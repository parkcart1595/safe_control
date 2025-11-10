import numpy as np
import cvxpy as cp

class BackupCBFQP:
    def __init__(self, robot, robot_spec, num_obs=10, kappa=10.0):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs
        self.kappa = kappa
        self.occlusion_scenarios = []
        
        self.sensing_range = float(self.robot_spec.get('sensing_range', 10.0))
        self.debug = bool(self.robot_spec.get('debug_backup_qp', False))

        # Backup CBF parameters
        self.T_horizon = 2.0   # backup time T
        self.dt_backup = 0.05   # backup trajectory sampling time step
        self.alpha = 1.0       # Class-K function

        self.setup_control_problem()
        
    def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None):
        """
        Compute the future trajectory (phi_b) and sensitivity matrix (Phi_b, STM) by following the backup controller from the current state x0.
        """
        from scipy.integrate import solve_ivp

        def augmented_dynamics(t, y):
            x = y[0:4]
            Phi = y[4:].reshape((4, 4))
            
            x_col = x.reshape(-1, 1)
            x_dot = self.f_cl(x_col, occlusion_scenarios).flatten()
            Phi_dot = self.F_cl(x) @ Phi
            
            # x_dot = self.f_cl(x.reshape(-1, 1)).flatten()
            # Phi_dot = self.F_cl(x) @ Phi
            
            return np.concatenate([x_dot, Phi_dot.flatten()])

        y0 = np.concatenate([x0.flatten(), np.eye(4).flatten()])
        t_eval = np.arange(0, T + dt, dt)
        
        sol = solve_ivp(
            augmented_dynamics,
            [0, T],
            y0,
            t_eval=t_eval,
            dense_output=True
        )
        
        backup_traj = sol.y[0:4, :].T       # (N,4)
        stm_traj = sol.y[4:, :].T.reshape(-1, 4, 4)
        
        return backup_traj, stm_traj, t_eval
        
    def _occlusion_barrier_softmax(self, pos, scenario, tau):
        """
        Compute soft-max occlusion barrier and its gradient at a backup state.

        Parameters
        ----------
        scenario : dict
            Contains:
                'A'         : (M, 2) half-space normals
                'b0'        : (M,) initial offsets
                'v_adv_max' : float, adversary speed bound
        tau : float
            Look-ahead time along the backup trajectory.

        Returns
        -------
        h_tilde : float or None
            Soft-max barrier value. None if invalid.
        grad_pos : np.ndarray or None, shape (1, 2)
            Gradient of h_tilde w.r.t. position. None if invalid.
        """
        A = scenario['A']      # (M,2)
        b0 = scenario['b0']    # (M,)
        v_adv = scenario['v_adv_max']
        R = self.robot_spec['radius']
        kappa = self.kappa

        if A.size == 0 or b0.size == 0:
            return None, None
        
        # b_i(τ) = b0_i + v_adv * tau
        b_tau = b0 + v_adv * tau  # (M,)

        # h_i = a_i^T pos - b_i(τ) - R
        # pos: (2,1) -> (M,)
        pos_flat = pos.reshape(2,)
        h_i = A @ pos_flat - b_tau - R   # (M,)
        
        if not np.all(np.isfinite(h_i)):
            return None, None

        M = h_i.shape[0]
        if M == 0:
            return None, None

        # Numerically stable log-sum-exp soft-max over {h_i}
        max_hi = np.max(h_i)
        z = np.exp(kappa * (h_i - max_hi))
        Z = np.sum(z)

        if not np.isfinite(Z) or Z <= 0.0:
            return None, None
        
        lse = max_hi + np.log(Z)
        h_tilde = (lse - np.log(M)) / kappa
        
        # softmax weights
        # w = np.exp(kappa * h_i)
        # w /= np.sum(w)
        # exp_shifted = np.exp(kappa * (h_i - max_hi))
        # w = exp_shifted / np.sum(exp_shifted)

        w = z / Z  # (M,)
        if not np.all(np.isfinite(w)):
            return None, None
        
        # grad wrt pos: sum_i w_i * a_i
        grad_pos = (w[:, None] * A).sum(axis=0, keepdims=True)  # (1,2)
        
        if not np.all(np.isfinite(grad_pos)):
            return None, None

        return float(h_tilde), grad_pos  # grad_pos (1,2)
    
    def _circle_tangents(self, p, c, R):
        """
        Compute tangent points from point p to a circle centered at c with radius R.

        t1, t2 :
            Tangent points on the circle (each (2,))
            if p is inside/on the circle (no valid tangents).
        """
        p = np.asarray(p, dtype=float).reshape(2,)
        c = np.asarray(c, dtype=float).reshape(2,)
        v = p - c
        d2 = float(v @ v)
        R2 = R * R

        # No tangent if p is inside or on the circle
        if d2 <= R2:
            return None, None

        x1, y1 = v
        x0 = R2 * x1 / d2
        y0 = R2 * y1 / d2
        k = R * np.sqrt(d2 - R2) / d2

        # Tangent points in global coordinates
        t1 = np.array([x0 - y1 * k, y0 + x1 * k]) + c
        t2 = np.array([x0 + y1 * k, y0 - x1 * k]) + c
        return t1, t2
    
    def _polygon_to_halfspaces(self, poly):
        """
        Convert a convex polygon into half-space form: { z | A z <= b }.
        """
        poly = np.asarray(poly, dtype=float)
        M = poly.shape[0]
        if M < 3:
            return None, None

        centroid = np.mean(poly, axis=0)

        A_list = []
        b_list = []

        for i in range(M):
            p1 = poly[i]
            p2 = poly[(i + 1) % M]
            edge = p2 - p1
            if np.linalg.norm(edge) < 1e-9:
                continue

             # Outward normal candidate (right-hand normal)
            n = np.array([edge[1], -edge[0]], dtype=float)
            n /= np.linalg.norm(n)
            
            b = float(n @ p1)

            # Ensure centroid is inside: n^T centroid <= b
            if n @ centroid > b + 1e-9:
                n = -n
                b = -b

            A_list.append(n)
            b_list.append(b)
            
        if len(A_list) == 0:
            return None, None

        A = np.vstack(A_list)        # (M_eff, 2)
        b0 = np.array(b_list)        # (M_eff,)
        
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b0))):
            return None, None
        
        return A, b0
    
    def _build_occlusion_scenario_for_obs(self, robot_state, obs):
        """
        Build an occlusion scenario for a single circular obstacle.

        The occlusion region is a wedge from the robot through the tangent points to a far range, then converted to half-spaces.

        scenario :
            {
              'A'         : (M_k, 2) half-space normals
              'b0'        : (M_k,) offsets
              'v_adv_max' : float, adversary speed bound
              'poly'      : (4, 2) occlusion polygon vertices
            }
            Returns None if no valid occlusion is formed.
        """

        px = float(robot_state[0, 0])
        py = float(robot_state[1, 0])
        p = np.array([px, py])

        obs = np.asarray(obs).flatten()
        ox, oy, r_obs = obs[:3]
        c = np.array([ox, oy])
        R_o = float(r_obs)

        sensing_R = self.sensing_range
        v_adv = float(self.robot_spec.get('v_adv_max_occ', 0.5))

        # Ignore obstacle if it is outside sensing range
        d = np.linalg.norm(c - p)
        if d >= sensing_R:
            return None

        # Compute tangent points from robot to obstacle
        t1, t2 = self._circle_tangents(p, c, R_o)
        if t1 is None or t2 is None:
            return None

        # Extend tangent directions to sensing range to form occlusion wedge
        dir1 = t1 - p
        n1 = np.linalg.norm(dir1)
        if n1 < 1e-6:
            return None
        dir1 /= n1
        
        dir2 = t2 - p
        n2 = np.linalg.norm(dir2)
        if n2 < 1e-6:
            return None
        dir2 /= n2

        far1 = p + sensing_R * dir1
        far2 = p + sensing_R * dir2

        # occlusion polygon: [t1, t2, far2, far1]
        poly = np.vstack([t1, t2, far2, far1])

        A, b0 = self._polygon_to_halfspaces(poly)
        if A is None:
            return None

        scenario = {
            'A': A,          # (M_k, 2)
            'b0': b0,        # (M_k,)
            'v_adv_max': v_adv,
            'poly': poly
        }
        return scenario

    def setup_control_problem(self):
        # QP variables and parameters
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))

        # max_constraints = 5 * int(self.T_horizon / self.dt_backup + 2)
        
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

        ######### Only detected obs version ########
        # if obs_list is None:
        #     self.status = 'optimal'
        #     return self.u_ref.value
        
        # if isinstance(obs_list, np.ndarray):
        #     if obs_list.size == 0:
        #         self.status = 'optimal'
        #         return self.u_ref.value

        # elif isinstance(obs_list, (list, tuple)):
        #     if len(obs_list) == 0:
        #         self.status = 'optimal'
        #         return self.u_ref.value
        
        # no obstacle and no occlusion => nominal
        # no_obs = (
        #     (obs_list is None) or
        #     (isinstance(obs_list, np.ndarray) and obs_list.size == 0) or
        #     (isinstance(obs_list, (list, tuple)) and len(obs_list) == 0)
        # )
        
        # 0) Filter obstacles within sensing range
        detected_obs = []
        
        if obs_list is not None:
            obs_arr = np.array(obs_list, dtype=float)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)

            px, py = float(robot_state[0, 0]), float(robot_state[1, 0])
            R_sense2 = self.sensing_range **2
            p = np.array([px, py])
            
            for obs in obs_arr:
                ox = float(obs[0])
                oy = float(obs[1])
                if (ox - px) ** 2 + (oy - py) ** 2 <= R_sense2:
                    detected_obs.append(obs)

        no_obs = (len(detected_obs) == 0)

        # 1) Build occlusion scenarios from detected obstacles
        occlusion_scenarios = []
        if not no_obs:
            for obs in detected_obs:
                scenario = self._build_occlusion_scenario_for_obs(robot_state, obs)
                if scenario is not None:
                    occlusion_scenarios.append(scenario)
        self.occlusion_scenarios = occlusion_scenarios
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

            for obs in detected_obs:
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
                    
            # for obs in obs_list:
            #     obs = np.asarray(obs).flatten()
            #     ox, oy, r_obs = obs[:3]

            #     if len(obs) >= 5:
            #         vx_o, vy_o = obs[3:5]
            #     else:
            #         vx_o, vy_o = 0.0, 0.0

            #     for i in range(1, len(tau_points)):  # tau = 0 제외
            #         tau = tau_points[i]

            #         phi_i = phi_b[i].reshape(-1, 1)   # (4,1)
            #         Phi_i = Phi_b[i]                  # (4,4)

            #         obs_pos_tau = np.array([
            #             [ox + vx_o * tau],
            #             [oy + vy_o * tau]
            #         ])

            #         # h^c(x, t+tau) = ||p(τ) - p_obs(τ)||^2 - d_min^2
            #         d_min = r_obs + self.robot_spec['radius']
            #         diff = phi_i[0:2] - obs_pos_tau            # (2,1)
            #         h_i = float(diff.T @ diff - d_min**2)

            #         # ∂h/∂phi = [2 (p - p_obs(τ))^T, 0, 0]
            #         grad_h_phi = 2.0 * np.hstack([diff.T, np.array([[0.0, 0.0]])])  # (1,4)

            #         # backup CBF chain rule
            #         f_x = self.robot.f(robot_state)      # (4,1)
            #         g_x = self.robot.g(robot_state)      # (4,2)

            #         Lfh_i = grad_h_phi @ Phi_i @ f_x     # (1,1)
            #         Lgh_i = grad_h_phi @ Phi_i @ g_x     # (1,2)

            #         # CBF inequality: L_f h + L_g h u + α h ≥ 0
            #         A_list.append(-Lgh_i)
            #         b_list.append(Lfh_i + self.alpha * h_i)
                    
        # 5) Occlusion-based adversarial constraints (backup CBF along trajectory)
        if not no_occ:
            for scenario in self.occlusion_scenarios:
                for i in range(1, len(tau_points)):
                    tau = tau_points[i]
                    phi_i = phi_b[i].reshape(-1, 1)
                    Phi_i = Phi_b[i]

                    pos_i = phi_i[0:2]  # (2,1)

                    h_tilde, grad_pos = self._occlusion_barrier_softmax(
                        pos_i, scenario, tau
                    )
                    # if h_tilde is None:
                    #     continue
                        
                    # if h_tilde < 0.0:
                    #     if self.debug:
                    #         print(f"[occ] skip at tau={tau:.2f}, h_tilde={h_tilde:.3e}<0")
                    #     continue
                    
                    if h_tilde is None or grad_pos is None:
                        continue

                    # Gradient w.r.t. backup state φ: [grad_pos, 0, 0]
                    grad_h_phi = np.hstack(
                        [grad_pos, np.array([[0.0, 0.0]])]
                    )  # (1,4)

                    # f_x = self.robot.f(robot_state)   # (4,1)
                    # g_x = self.robot.g(robot_state)   # (4,2)

                    Lfh = grad_h_phi @ Phi_i @ f_x    # (1,1)
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

                    # if (np.all(np.isfinite(Lgh)) and
                    #     np.all(np.isfinite(Lfh)) and
                    #     np.isfinite(h_tilde)):
                    #     A_list.append(-Lgh)
                    #     b_list.append(Lfh + self.alpha * h_tilde)
                    
        # 6) Backup set constraint at final time
        phi_T = phi_b[-1].reshape(-1, 1)
        Phi_T = Phi_b[-1]

        h_b_T = self.robot.h_b_stop(phi_T)
        grad_h_b_T = self.robot.grad_h_b_stop(phi_T)

        Lfh_b_T = grad_h_b_T @ Phi_T @ self.robot.f(robot_state)
        Lgh_b_T = grad_h_b_T @ Phi_T @ self.robot.g(robot_state)

        if np.all(np.isfinite(Lgh_b_T)) and np.all(np.isfinite(Lfh_b_T)) and np.all(np.isfinite(h_b_T)):
            # A_list.append(-Lgh_b_T)
            # b_list.append(Lfh_b_T + self.alpha * h_b_T)
            rhs_b = float(Lfh_b_T + self.alpha * h_b_T)

            # Skip degenerate constraint that would be infeasible by construction
            if np.linalg.norm(Lgh_b_T) < 1e-9 and rhs_b < 0.0:
                print("loop1 in")
                if self.debug:
                    print(f"[backup] skip degenerate (||Lgh||≈0, rhs={rhs_b:.3e}<0)")
            else:
                print("loop2 in")
                A_list.append(-Lgh_b_T)
                b_list.append(np.array([[rhs_b]]))

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

        # 8) Solve QP (try GUROBI first, fall back to OSQP)
        try:
            self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
        except cp.error.SolverError:
            self.cbf_controller.solve(solver=cp.OSQP)
            
        self.status = self.cbf_controller.status

        if self.status != 'optimal':
            print("loop3 in")
            if (not no_occ) and hasattr(self.robot, "backup_input_occlusion"):
                print("loop4 in")
                return self.robot.backup_input_occlusion(robot_state, self.occlusion_scenarios)
            elif hasattr(self.robot, "backup_input"):
                print("loop5 in")
                return self.robot.backup_input(robot_state)
            elif hasattr(self.robot, "stop"):
                print("loop6 in")
                return self.robot.stop(robot_state)
            else:
                print("loop7 in")
                return np.zeros((2, 1))
            
        # if self.status != 'optimal':
        #     return (self.robot.backup_input_occlusion(robot_state, self.occlusion_scenarios)
        #         if not no_occ else
        #         self.robot.backup_input(robot_state))
        #     # return self.robot.backup_input(robot_state)

        return self.u.value

    # def solve_control_problem(self, robot_state, control_ref, obs_list):
    #     self.u_ref.value = control_ref['u_ref']
        
    #     if obs_list is None:
    #         # if no obstacle, use u_ref
    #         self.status = 'optimal'
    #         return self.u_ref.value

    #     A_list, b_list = [], []
        
    #     phi_b, Phi_b, tau_points = self.robot.simulate_backup_trajectory(
    #         robot_state, self.T_horizon, self.dt_backup
    #     )
        
    #     # 1. obsatcle avoidance constraints (trajectory constraint)
    #     for obs in obs_list:

    #         for i in range(1, len(tau_points)): # except tau=0
    #             phi_i = phi_b[i].reshape(-1, 1)
    #             Phi_i = Phi_b[i]

    #             # h(x) = ||x_pos - obs_pos||^2 - d_min^2
    #             d_min = obs[2] + self.robot_spec['radius']
    #             h_i = np.linalg.norm(phi_i[0:2] - obs[0:2].reshape(-1, 1))**2 - d_min**2
                
    #             pos_diff_col = phi_i[0:2] - obs[0:2].reshape(-1, 1) # (2,1) - (2,1) = (2,1)
    #             grad_h_i = 2 * np.hstack([pos_diff_col.T, np.array([[0, 0]])])

    #             # calculate Lie Derivative)
    #             # h_dot = grad_h @ (Phi @ (f(x) + g(x)u))
    #             Lfh_i = grad_h_i @ Phi_i @ self.robot.f(robot_state)
    #             Lgh_i = grad_h_i @ Phi_i @ self.robot.g(robot_state)

    #             # QP: A*u <= b
    #             A_list.append(-Lgh_i)
    #             b_list.append(Lfh_i + self.alpha * h_i)

    #     # 2. Backup set constraint (final state constraint)
    #     # final state phi_T should reach backup set
    #     phi_T = phi_b[-1].reshape(-1, 1)
    #     Phi_T = Phi_b[-1]
        
    #     h_b_T = self.robot.h_b_stop(phi_T)
    #     grad_h_b_T = self.robot.grad_h_b_stop(phi_T)

    #     Lfh_b_T = grad_h_b_T @ Phi_T @ self.robot.f(robot_state)
    #     Lgh_b_T = grad_h_b_T @ Phi_T @ self.robot.g(robot_state)
    #     A_list.append(-Lgh_b_T)
    #     b_list.append(Lfh_b_T + self.alpha * h_b_T)

    #     # 3. QP parameter update and solve
    #     num_constraints = len(A_list)
    #     A_cbf_val = np.array(A_list).reshape(num_constraints, 2)
    #     b_cbf_val = np.array(b_list).reshape(num_constraints, 1)

    #     self.A_cbf.value[:, :] = 0
    #     self.A_cbf.value[:num_constraints, :] = A_cbf_val
    #     self.b_cbf.value[:, :] = 1e6
    #     self.b_cbf.value[:num_constraints, :] = b_cbf_val
        
    #     self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
    #     self.status = self.cbf_controller.status
        
    #     if self.status != 'optimal':
    #         # if qp is not solved, use backup policy
    #         return self.robot.backup_input(robot_state)
            
    #     return self.u.value