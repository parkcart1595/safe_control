import numpy as np
import cvxpy as cp
import time

from utils.occlusion import OcclusionUtils

class InfeasibleError(Exception):
    '''Raised when the QP is infeasible or a collision is detected.'''

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
        # Backup CBF parameters (override via robot_spec["backup_cbf"]).
        self.T_horizon = float(cfg.get('T_horizon', 3.0))   # backup time T
        self.dt_backup = float(cfg.get('dt_backup', 0.05))  # backup trajectory sampling time step
        self.alpha = float(cfg.get('alpha', 2.0))           # Class-K function

        OcclusionUtils.__init__(
            self,
            robot=robot,
            robot_spec=robot_spec,
            sensing_range=self.sensing_range,
            barrier_fn=self._occlusion_barrier_smax_curved,
        )

        self.setup_control_problem()
        
        target = self.robot
        try:
            if hasattr(target, "set_occ_barrier_fn"):
                target.set_occ_barrier_fn(self._occlusion_barrier_smax_curved)
            elif hasattr(target, "robot") and hasattr(target.robot, "set_occ_barrier_fn"):
                target.robot.set_occ_barrier_fn(self._occlusion_barrier_smax_curved)
            else:
                print("[BackupCBFQP][WARN] Could not inject occlusion barrier callback "
                    "(no set_occ_barrier_fn on robot).")
        except Exception as e:
            print(f"[BackupCBFQP][WARN] Failed to inject occ barrier callback: {e}")
    
    def _occlusion_barrier_smax_curved(self, pos, scenario, tau=0.0):
        """
        smooth-max occlusion barrier built from two tangent half-spaces and
        an expanded obstacle arc. Returns (h_tilde, grad_pos, risk_normal_vec).
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

        if 'v_expand_vec' in scenario:
            v_expand = scenario['v_expand_vec']
        else:
            v_expand = pi_adv
        # --- expanded obstacle disk ---
        R_occ =  R + v_expand * float(tau)

        # primitives: phi_k(p) = a_k^T p - b_k - delta  (inside inflated poly => <=0)
        h_vec = (A @ pos) - b0 - R_occ    # (K,)
        if not np.all(np.isfinite(h_vec)):
            return None, None, None

        # Smoth-max via log-sum-exp
        max_h = np.max(h_vec)
        z = np.exp(kappa * (h_vec - max_h))
        Z = np.sum(z)
        if not np.isfinite(Z) or Z <= 0.0:
            return None, None, None
        
        # smooth-max value
        K = h_vec.size
        h_tilde = (max_h + np.log(Z) - np.log(K)) / kappa

        # # gradient: sum_k lambda_k * a_k
        # lam = z / Z
        # h_tilde = (max_h + np.log(Z) - np.log(K)) / kappa

        # gradient
        lam = z / Z
        grad_pos = (lam[:, None] * A).sum(axis=0, keepdims=True)

        # for debug/visualization: normals of "active-ish" facets
        active = (h_vec >= 0.0)
        risk_normal_vec = A[active] if np.any(active) else np.empty((0,2))

        return float(h_tilde), grad_pos, risk_normal_vec
    
    def setup_control_problem(self):
        # QP variables and parameters
        u_dim = int(getattr(self.robot, "u_dim", self.robot_spec.get("u_dim", 2)))

        self.u = cp.Variable((u_dim, 1))
        self.u_ref = cp.Parameter((u_dim, 1), value=np.zeros((u_dim, 1)))

        N_tau = int(self.T_horizon / self.dt_backup) + 2
        max_constraints = int((2 * self.num_obs + 10) * N_tau)
        
        self.A_cbf = cp.Parameter((max_constraints, u_dim), value=np.zeros((max_constraints, u_dim)))
        self.b_cbf = cp.Parameter((max_constraints, 1), value=np.zeros((max_constraints, 1)))
        
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
        
        constraints = [self.A_cbf @ self.u <= self.b_cbf]

        ic_fn = getattr(self.robot, "input_constraints", None)
        if callable(ic_fn):
            constraints.extend(ic_fn(self.u))
        else:
            if u_dim == 2 and 'a_max' in self.robot_spec:
                constraints.extend([
                    cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                    cp.abs(self.u[1]) <= self.robot_spec['a_max']
                ])

        self.cbf_controller = cp.Problem(objective, constraints)
        self.status = None
        
    def set_occlusion_scenarios(self, scenarios):
        """
        Set occlusion-based adversarial scenarios.
        """
        self.occlusion_scenarios = scenarios
        
    def solve_control_problem(self, robot_state, control_ref, obs_list):
        self.u_ref.value = control_ref['u_ref']
        
        # 1) Build visible obstacles + occlusion scenarios (sensing range handled inside)
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

        # 3) Compute backup trajectory under the backup policy
        phi_b, Phi_b, tau_points = self.robot.simulate_backup_trajectory(
            robot_state, self.T_horizon, self.dt_backup,
            occlusion_scenarios=None if no_occ else self.occlusion_scenarios
        )

        # Pre-compute f(x), g(x) at current state for Lie derivatives
        f_x = self.robot.f(robot_state)   # (n,1)
        g_x = self.robot.g(robot_state)   # (n,u)
        
        # 4) Use HOCBF constraints for visible obstacles
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
                # Relative degree 2 HOCBF
                h_dot = 2.0 * float(p_rel @ v_rel)

                # h_ddot = 2||v_rel||^2 + 2 p_rel^T u
                v_rel_norm2 = float(v_rel @ v_rel)

                psi1 = h_dot + gamma1 * h

                A = -2.0 * p_rel.reshape(1, 2)  # (1,2)
                b = 2.0 * v_rel_norm2 + gamma2 * psi1  # scalar

                if np.all(np.isfinite(A)) and np.isfinite(b):
                    A_list.append(A)
                    b_list.append(np.array([[b]]))
                    
        # 5) Occlusion constraints along the backup trajectory
        if not no_occ:
            for scenario in self.occlusion_scenarios:
                for i in range(1, len(tau_points)):
                    tau = tau_points[i]
                    phi_i = phi_b[i].reshape(-1, 1)
                    Phi_i = Phi_b[i]

                    pos_i = phi_i[0:2]  # (2,1)

                    h_tilde, grad_pos, _ = self._occlusion_barrier_smax_curved(
                        pos_i, scenario, tau
                    )
                    
                    if h_tilde is None or grad_pos is None:
                        continue

                    if 'v_expand_vec' in scenario:
                        v_exp = scenario['v_expand_vec']
                    else:
                        v_exp = np.full(len(scenario['b0']), scenario.get('v_adv_max', 0.5))

                    R = self.robot_spec['radius']
                    A_sc = scenario['A']
                    b0_sc = scenario['b0']
                    
                    delta_vec = v_exp * float(tau)
                    
                    # h_vec = A p - b0 - delta - R
                    h_vec = (A_sc @ pos_i.flatten()) - b0_sc - delta_vec - R
                    
                    # Log-Sum-Exp 
                    max_h = np.max(h_vec)
                    z = np.exp(self.kappa * (h_vec - max_h))
                    Z = np.sum(z)
                    lam = z / Z  # shape: (K,)

                    dh_dt = np.dot(lam, -v_exp)

                    # Pad gradient to the full backup state dimension
                    state_dim = Phi_i.shape[0]
                    pad_cols = max(0, state_dim - grad_pos.shape[1])
                    grad_h_phi = np.hstack([grad_pos, np.zeros((1, pad_cols))])
                    
                    u_pi_phi = self._u_pi_at(phi_i, self.occlusion_scenarios, t=tau)
                    u_pi_phi = np.asarray(u_pi_phi, dtype=float).reshape(2,1)
                    f_pi_phi = self.robot.f(phi_i) + self.robot.g(phi_i) @ u_pi_phi

                    # Pad f/g if needed to match Phi_i dimension
                    if f_pi_phi.shape[0] < state_dim:
                        f_pi_phi = np.vstack([f_pi_phi, np.zeros((state_dim - f_pi_phi.shape[0], 1))])
                    if f_x.shape[0] < state_dim:
                        f_x_pad = np.vstack([f_x, np.zeros((state_dim - f_x.shape[0], 1))])
                    else:
                        f_x_pad = f_x
                    if g_x.shape[0] < state_dim:
                        g_x_pad = np.vstack([g_x, np.zeros((state_dim - g_x.shape[0], g_x.shape[1]))])
                    else:
                        g_x_pad = g_x

                    Lfh = grad_h_phi @ (Phi_i @ f_x_pad)     # (1,1)
                    Lgh = grad_h_phi @ Phi_i @ g_x_pad                  # (1,2)
                    
                    if not (np.all(np.isfinite(Lgh)) and np.all(np.isfinite(Lfh))):
                        continue

                    rhs = float(Lfh + dh_dt + self.alpha * h_tilde)

                    # If control has almost no effect but rhs < 0, skip this
                    if np.linalg.norm(Lgh) < 1e-9 and rhs < 0.0:
                        if self.debug:
                            print(f"[occ] skip degenerate (||Lgh||≈0, rhs={rhs:.3e}<0)")
                        continue
                    
                    # Standard CBF inequality: -L_g h u ≤ L_f h + α h
                    A_list.append(-Lgh)
                    b_list.append(np.array([[rhs]]))
                    
        # 6) Terminal backup set constraint
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

        state_dim_T = Phi_T.shape[0]
        f_term = self.robot.f(robot_state)
        g_term = self.robot.g(robot_state)
        if f_term.shape[0] < state_dim_T:
            f_term = np.vstack([f_term, np.zeros((state_dim_T - f_term.shape[0], 1))])
        if g_term.shape[0] < state_dim_T:
            g_term = np.vstack([g_term, np.zeros((state_dim_T - g_term.shape[0], g_term.shape[1]))])

        u_pi_T = self._u_pi_at(phi_T, self.occlusion_scenarios, t=self.T_horizon)
        f_pi_T = self.robot.f(phi_T) + self.robot.g(phi_T) @ u_pi_T

        Lfh_b_T = grad_h_b_T @ (Phi_T @ f_term)
        Lgh_b_T = grad_h_b_T @ Phi_T @ g_term
        rhs_b   = float(Lfh_b_T + self.alpha * h_b_T)
        A_list.append(-Lgh_b_T)
        b_list.append([[rhs_b]])

        # Lfh_b_T = grad_h_b_T @ Phi_T @ f_term
        # Lgh_b_T = grad_h_b_T @ Phi_T @ g_term

        # if np.all(np.isfinite(Lgh_b_T)) and np.all(np.isfinite(Lfh_b_T)) and np.all(np.isfinite(h_b_T)):
        #     # A_list.append(-Lgh_b_T)
        #     # b_list.append(Lfh_b_T + self.alpha * h_b_T)
        #     rhs_b = float(Lfh_b_T + self.alpha * h_b_T)

        #     # Skip degenerate constraint that is infeasible by construction
        #     if np.linalg.norm(Lgh_b_T) < 1e-9 and rhs_b < 0.0:
        #         if self.debug:
        #             print(f"[backup] skip degenerate (||Lgh||≈0, rhs={rhs_b:.3e}<0)")
        #     else:
        #         # print("loop_feasible in")
        #         A_list.append(-Lgh_b_T)
        #         b_list.append(np.array([[rhs_b]]))

        # 7) If no constraints, use nominal control
        num_constraints = len(A_list)

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
