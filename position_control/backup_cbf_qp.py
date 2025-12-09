import numpy as np
import cvxpy as cp
import time

from utils.occlusion import OcclusionUtils

class InfeasibleError(Exception):
    '''
    Exception raised for errors when QP is infeasible or 
    the robot collides with the obstacle
    '''

    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)

class BackupCBFQP(OcclusionUtils):
    def __init__(self, robot, robot_spec, num_obs=10, kappa=10.0):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs
        self.kappa = kappa
        self.occlusion_scenarios = []
        
        self.sensing_range = float(self.robot_spec.get('sensing_range', 10.0))
        self.debug = bool(self.robot_spec.get('debug_backup_qp', False))

        cfg = self.robot_spec.get('backup_cbf', {})
        # Backup CBF parameters (override-able via robot_spec["backup_cbf"])
        self.T_horizon = float(cfg.get('T_horizon', 3.0))   # backup time T
        self.dt_backup = float(cfg.get('dt_backup', 0.05))  # backup trajectory sampling time step
        self.alpha = float(cfg.get('alpha', 2.0))           # Class-K function

        OcclusionUtils.__init__(
            self,
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=self._occlusion_barrier_softmax_curved,
        )

        self.setup_control_problem()
        
        target = self.robot
        try:
            if hasattr(target, "set_occ_barrier_fn"):
                target.set_occ_barrier_fn(self._occlusion_barrier_softmax_curved)
            elif hasattr(target, "robot") and hasattr(target.robot, "set_occ_barrier_fn"):
                target.robot.set_occ_barrier_fn(self._occlusion_barrier_softmax_curved)
            else:
                print("[BackupCBFQP][WARN] Could not inject occlusion barrier callback "
                    "(no set_occ_barrier_fn on robot).")
        except Exception as e:
            print(f"[BackupCBFQP][WARN] Failed to inject occ barrier callback: {e}")
    
    def _occlusion_barrier_softmax_curved(self, pos, scenario, tau=0.0):
        """
        Curved occlusion barrier:
        - h1, h2 : two tangent half-space
        - h3     : sensing range arc
        - h4     : obstacle arc

        h_tilde(x) = (1/kappa) [ log Σ exp(kappa h_i(x)) - log(4) ]
        """
        p = scenario['robot_pos']       # robot position
        c = scenario['obs_center']      # obstacle center
        R_o = scenario['obs_radius']    # obstacle radius
        pi_adv = scenario['v_adv_max']
        R = self.robot_spec['radius']   # robot radius (margin)
        kappa = self.kappa
        
        A = scenario.get('A', None)
        b0 = scenario.get('b0', None)
        if A is None or b0 is None or A.shape[0] < 2:
            return None, None, None
        
        pos = np.asarray(pos, float).reshape(2,)
        x, y = pos

        R_occ = pi_adv * float(tau)
        # --- tangent halfspaces (use first 2 rows) ---
        a1, a2 = A[3], A[1]
        beta1, beta2 = b0[3], b0[1]

        # safe if outside wedge (plus robot radius margin)
        h1 = a1 @ pos - beta1 - R - R_occ
        h2 = a2 @ pos - beta2 - R - R_occ
        
        A_stack = []
        if h1 >=0:
            A_stack.append(a1)
        if h2 >=0:
            A_stack.append(a2)
            
        if A_stack:
            risk_normal_vec = np.vstack(A_stack)
        else:
            risk_normal_vec = np.empty((0,2))

        # --- expanded obstacle disk ---
        R_occ = pi_adv * float(tau)
        dx_o = x - c[0]
        dy_o = y - c[1]
        # safe if outside expanded disk (plus robot radius)
        # h3 = np.sqrt(dx_o*dx_o + dy_o*dy_o - (R_occ + R + R_o)**2)
        d = np.hypot(dx_o, dy_o)
        R_tot = R + R_o + R_occ
        h3 = d - R_tot
        # print(h3)

        h_i = np.array([h1, h2, h3], dtype=float)
        # print(h_i)
        if not np.all(np.isfinite(h_i)):
            return None, None, None

        # log-sum-exp soft-max
        M = h_i.size
        max_h = np.max(h_i)
        # print(max_h)
        z = np.exp(kappa * (h_i - max_h))
        Z = np.sum(z)
        if not np.isfinite(Z) or Z <= 0.0:
            return None, None, None

        lse = max_h + np.log(Z)
        h_tilde = (lse - np.log(M)) / kappa
        # print(h_tilde)
        # gradient:
        # ∂h1/∂x = a1
        # ∂h2/∂x = a2
        # ∂h3/∂x = 2(x-p)
        # ∂h4/∂x = -2(x-c)
        dh1 = a1
        dh2 = a2
        # dh3 = np.array([2.0 * dx_s, 2.0 * dy_s])
        # dh3 = 2.0 * np.array([dx_o, dy_o])
        eps = 1e-9
        # dh3 = (1.0 / (h3 + eps)) * np.array([dx_o, dy_o])
        dh3 = np.array([dx_o, dy_o]) / max(eps, d)
        

        grads = np.vstack([dh1, dh2, dh3])  # (4,2)

        lambda_i = z / Z
        grad_pos = (lambda_i[:, None] * grads).sum(axis=0, keepdims=True)

        if not np.all(np.isfinite(grad_pos)):
            return None, None, None
        
        return float(h_tilde), grad_pos, risk_normal_vec
    
    def setup_control_problem(self):
        # QP variables and parameters
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))

        N_tau = int(self.T_horizon / self.dt_backup) + 2
        max_constraints = int((2 * self.num_obs + 10) * N_tau)
        
        self.A_cbf = cp.Parameter((max_constraints, 2), value=np.zeros((max_constraints, 2)))
        self.b_cbf = cp.Parameter((max_constraints, 1), value=np.zeros((max_constraints, 1)))
        
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
        
        constraints = [
            self.A_cbf @ self.u <= self.b_cbf,
            cp.abs(self.u[0]) <= self.robot_spec['a_max'],
            cp.abs(self.u[1]) <= self.robot_spec['a_max']
        ]

        self.cbf_controller = cp.Problem(objective, constraints)
        self.status = None
        
    def set_occlusion_scenarios(self, scenarios):
        """
        Set occlusion-based adversarial scenarios.
        """
        self.occlusion_scenarios = scenarios
        
    def solve_control_problem(self, robot_state, control_ref, obs_list):
        self.u_ref.value = control_ref['u_ref']
        
        # 0) Build visible obstacles + occlusion scenarios (sensing range handled inside)
        visible_obs, occlusion_scenarios = self._filter_visible_and_build_occ(robot_state, obs_list)
        self.occlusion_scenarios = occlusion_scenarios
        no_obs = (len(visible_obs) == 0)
        no_occ = (len(self.occlusion_scenarios) == 0)

        # 2) If there is no obstacle and no occlusion, use nominal control
        if no_obs and no_occ:
            self.status = 'optimal'
            if self.debug:
                print("[BackupCBFQP] no detected obstacle/occlusion -> use u_ref")
            return self.u_ref.value

        A_list, b_list = [], []

        # 3) Compute backup trajectory under pi_backup
        phi_b, Phi_b, tau_points = self.robot.simulate_backup_trajectory(
            robot_state, self.T_horizon, self.dt_backup,
            occlusion_scenarios=None if no_occ else self.occlusion_scenarios
        )

        # Pre-compute f(x), g(x) at current state for Lie derivatives
        f_x = self.robot.f(robot_state)   # (4,1)
        g_x = self.robot.g(robot_state)   # (4,2)
        
        # 4) Safety constraints for detected obstacles (instantaneous HOCBF)
        if not no_obs:
            x = float(robot_state[0, 0])
            y = float(robot_state[1, 0])
            vx = float(robot_state[2, 0])
            vy = float(robot_state[3, 0])
            p = np.array([x, y])
            v = np.array([vx, vy])

            R_robot = self.robot_spec['radius']
            gamma1 = 1.0  # class k_1
            gamma2 = 1.0  # class k_2

            for obs in visible_obs:
                obs = np.asarray(obs, dtype=float).flatten()
                ox, oy, r_obs = obs[:3]
                if len(obs) >= 5:
                    vx_o, vy_o = obs[3:5]
                else:
                    vx_o, vy_o = 0.0, 0.0

                p_obs = np.array([ox, oy])
                v_obs = np.array([vx_o, vy_o])

                p_rel = p - p_obs
                v_rel = v - v_obs

                d_min = r_obs + R_robot

                h = float(p_rel @ p_rel - d_min**2)
                # relative degree 2 HOCBF
                h_dot = 2.0 * float(p_rel @ v_rel)

                # h_ddot = 2||v_rel||^2 + 2 p_rel^T u
                v_rel_norm2 = float(v_rel @ v_rel)

                psi1 = h_dot + gamma1 * h

                A = -2.0 * p_rel.reshape(1, 2)  # (1,2)
                b = 2.0 * v_rel_norm2 + gamma2 * psi1  # scalar

                if np.all(np.isfinite(A)) and np.isfinite(b):
                    A_list.append(A)
                    b_list.append(np.array([[b]]))
                    
        # 5) Occlusion-based adversarial constraints (backup CBF along trajectory)
        if not no_occ:
            for scenario in self.occlusion_scenarios:
                for i in range(1, len(tau_points)):
                    tau = tau_points[i]
                    phi_i = phi_b[i].reshape(-1, 1)
                    Phi_i = Phi_b[i]

                    pos_i = phi_i[0:2]  # (2,1)
                    # print(f"pos_i : {pos_i}")

                    h_tilde, grad_pos, _ = self._occlusion_barrier_softmax_curved(
                        pos_i, scenario, tau
                    )
                    
                    if h_tilde is None or grad_pos is None:
                        continue

                    # Gradient w.r.t. backup state φ: [grad_pos, 0, 0]
                    grad_h_phi = np.hstack(
                        [grad_pos, np.array([[0.0, 0.0]])]
                    )  # (1,4)
                    
                    u_pi_phi = self._u_pi_at(phi_i, self.occlusion_scenarios, t=tau)
                    u_pi_phi = np.asarray(u_pi_phi, dtype=float).reshape(2,1)
                    f_pi_phi = self.robot.f(phi_i) + self.robot.g(phi_i) @ u_pi_phi

                    Lfh = grad_h_phi @ (Phi_i @ f_x - f_pi_phi)    # (1,1)
                    Lgh = grad_h_phi @ Phi_i @ g_x    # (1,2)
                    
                    if not (np.all(np.isfinite(Lgh)) and np.all(np.isfinite(Lfh))):
                        continue

                    rhs = float(Lfh + self.alpha * h_tilde)

                    # If control has almost no effect but rhs < 0, skip this
                    if np.linalg.norm(Lgh) < 1e-9 and rhs < 0.0:
                        if self.debug:
                            print(f"[occ] skip degenerate (||Lgh||≈0, rhs={rhs:.3e}<0)")
                        continue
                    
                    # CBF form: L_f h + L_g h u + α h ≥ 0  →  -L_g h u ≤ L_f h + α h
                    A_list.append(-Lgh)
                    b_list.append(np.array([[rhs]]))
                    
        # 6) Backup set constraint at final time
        phi_T = phi_b[-1].reshape(-1, 1)
        Phi_T = Phi_b[-1]
        
        term_scn = None if no_occ else self.occlusion_scenarios[0]
        self.robot.set_terminal_backup_context(
            term_scn,
            self.T_horizon,
            kappa=self.kappa,
            rho_T=0.1
        )

        h_b_T = self.robot.h_b_stop(phi_T)
        grad_h_b_T = self.robot.grad_h_b_stop(phi_T)
        # print(f"h_b_T: {h_b_T}, grad_h_b_T: {grad_h_b_T}")

        Lfh_b_T = grad_h_b_T @ Phi_T @ self.robot.f(robot_state)
        Lgh_b_T = grad_h_b_T @ Phi_T @ self.robot.g(robot_state)

        if np.all(np.isfinite(Lgh_b_T)) and np.all(np.isfinite(Lfh_b_T)) and np.all(np.isfinite(h_b_T)):
            # A_list.append(-Lgh_b_T)
            # b_list.append(Lfh_b_T + self.alpha * h_b_T)
            rhs_b = float(Lfh_b_T + self.alpha * h_b_T)

            # Skip degenerate constraint that would be infeasible by construction
            if np.linalg.norm(Lgh_b_T) < 1e-9 and rhs_b < 0.0:
                if self.debug:
                    print(f"[backup] skip degenerate (||Lgh||≈0, rhs={rhs_b:.3e}<0)")
            else:
                # print("loop_feasible in")
                A_list.append(-Lgh_b_T)
                b_list.append(np.array([[rhs_b]]))

        # 7) If no constraints, use nominal control
        num_constraints = len(A_list)
        # print(f"num_cons: {num_constraints}")
        if num_constraints == 0:
            self.status = 'optimal'
            if self.debug:
                print("[BackupCBFQP] no constraints -> use u_ref")
            return self.u_ref.value

        # Stack constraints into parameter matrices
        A_cbf_val = np.vstack(A_list).reshape(num_constraints, 2)
        b_cbf_val = np.vstack(b_list).reshape(num_constraints, 1)

        self.A_cbf.value[:, :] = 0.0
        self.b_cbf.value[:, :] = 1e6
        self.A_cbf.value[:num_constraints, :] = A_cbf_val
        self.b_cbf.value[:num_constraints, :] = b_cbf_val
        
        if self.debug:
            viol = float(np.max(A_cbf_val @ self.u_ref.value - b_cbf_val))
            print(f"[BackupCBFQP] num_constraints={num_constraints}, max_viol(u_ref)={viol:.3e}")

        # 8) Solve QP (GUROBI only; status is handled by tracking.py)
        t_start_qp = time.perf_counter()
        self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
        t_end_qp = time.perf_counter()

        qp_solve_time_ms = (t_end_qp - t_start_qp) * 1000

        if self.debug:
            print(f"[BackupCBFQP] QP Solve Time: {qp_solve_time_ms:.3f} ms")

        self.status = self.cbf_controller.status

        return self.u.value