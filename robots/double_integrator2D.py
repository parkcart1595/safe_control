import numpy as np
import casadi as ca

"""
Created on July 15th, 2024
@author: Taekyung Kim

@description: 
Double Integrator model for CBF-QP and MPC-CBF (casadi) with separated position and attitude states
"""


def angle_normalize(x):
    if isinstance(x, (np.ndarray, float, int)):
        # NumPy implementation
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        # CasADi implementation
        return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


class DoubleIntegrator2D:

    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, vx, vy]
            theta: yaw angle
            U: [ax, ay]
            U_attitude: [yaw_rate]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
            relative degree: 2
        '''
        self.dt = dt
        self.robot_spec = robot_spec

        self.robot_spec.setdefault('a_max', 1.0)
        self.robot_spec.setdefault('v_max', 1.0)
        self.robot_spec.setdefault('ax_max', self.robot_spec['a_max'])
        self.robot_spec.setdefault('ay_max', self.robot_spec['a_max'])
        self.robot_spec.setdefault('w_max', 0.5)
        
        self.pid_occ = {
            "I": np.zeros(2),      # integral of velocity error
            "e_prev": np.zeros(2)  # previous velocity error (for D term)
        }
        
        self.pid_occ_gains = {
            "Kp": 1.0,
            "Ki": 0.2,
            "Kd": 0.1,
            "aw_limit": 1.0
        }
        cfg = self.robot_spec.setdefault("backup_cbf", {})
        self.T_horizon = float(cfg.get("T_horizon", 2.0))
        cfg["T_horizon"] = self.T_horizon

        # controller config
        self.u_dim = 2

    def input_constraints(self, u_var):
        """Return CVXPY constraints for input bounds."""
        import cvxpy as cp
        a_max = float(self.robot_spec.get('a_max', np.inf))
        return [cp.abs(u_var[0]) <= a_max,
                cp.abs(u_var[1]) <= a_max]

    def f(self, X, casadi=False):
        if casadi:
            return ca.vertcat(
                X[2, 0],
                X[3, 0],
                0,
                0
            )
        else:
            return np.array([X[2, 0],
                             X[3, 0],
                             0,
                             0]).reshape(-1, 1)

    def df_dx(self, X):
        return np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def g(self, X, casadi=False):
        if casadi:
            return ca.DM([
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1]
            ])
        else:
            return np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

    def step(self, X, U):
        # X = X + (self.f(X) + self.g(X) @ U) * self.dt
        # return X
        X_next = X + (self.f(X) + self.g(X) @ U) * self.dt
        v_max = float(self.robot_spec.get('v_max', np.inf))
        vx, vy = float(X_next[2,0]), float(X_next[3,0])
        vnorm = (vx**2 + vy**2)**0.5
        if vnorm > v_max and np.isfinite(v_max):
            scale = v_max / vnorm
            X_next[2,0] *= scale
            X_next[3,0] *= scale
        return X_next

    def step_rotate(self, theta, U_attitude):
        theta = angle_normalize(theta + U_attitude[0, 0] * self.dt)
        return theta

    def nominal_input(self, X, G, d_min=0.05, k_v=1.0, k_a=1.0):
        '''
        nominal input for CBF-QP (position control)
        '''
        G = np.copy(G.reshape(-1, 1))  # goal state
        v_max = self.robot_spec['v_max']  # Maximum velocity (x+y)
        a_max = self.robot_spec['a_max']  # Maximum acceleration

        pos_errors = G[0:2, 0] - X[0:2, 0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)

        # Compute desired velocities for x and y
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag

        # Compute accelerations
        current_v = X[2:4, 0]
        a = k_a * (v_des - current_v)
        a_mag = np.linalg.norm(a)
        if a_mag > a_max:
            a = a * a_max / a_mag

        return a.reshape(-1, 1)

    def nominal_attitude_input(self, theta, theta_des, k_theta=1.0):
        '''
        nominal input for attitude control
        '''
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_theta * error_theta
        return np.array([yaw_rate]).reshape(-1, 1)

    def stop(self, X, k_a=1.0):
        # Set desired velocity to zero
        vx_des, vy_des = 0.0, 0.0
        ax = k_a * (vx_des - X[2, 0])
        ay = k_a * (vy_des - X[3, 0])
        return np.array([ax, ay]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        return np.linalg.norm(X[2:4, 0]) < tol

    def rotate_to(self, theta, theta_des, k_omega=2.0):
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_omega * error_theta
        yaw_rate = np.clip(yaw_rate, -self.robot_spec['w_max'], self.robot_spec['w_max'])
        return np.array([yaw_rate]).reshape(-1, 1)

    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''Continuous Time High Order CBF'''
        obsX = obs[0:2].reshape(-1, 1)
        d_min = obs[2] + robot_radius  # obs radius + robot radius

        h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta*d_min**2
        # Lgh is zero => relative degree is 2, f(x)[0:2] actually equals to X[2:4]
        h_dot = 2 * (X[0:2] - obsX[0:2]).T @ (self.f(X)[0:2])

        # these two options are the same
        # df_dx = self.df_dx(X)
        # dh_dot_dx = np.append( ( 2 * self.f(X)[0:2] ).T, np.array([[0,0]]), axis = 1 ) + 2 * ( X[0:2] - obsX[0:2] ).T @ df_dx[0:2,:]
        dh_dot_dx = np.append(2 * X[2:4].T, 2 * (X[0:2] - obsX[0:2]).T, axis=1)
        return h, h_dot, dh_dot_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

        def h(x, obs, robot_radius, beta=1.01):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            y_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 - beta*d_min**2
            return h

        h_k2 = h(x_k2, obs, robot_radius, beta)
        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        dd_h = h_k2 - 2 * h_k1 + h_k
        # hocbf_2nd_order = h_ddot + (gamma1 + gamma2) * h_dot + (gamma1 * gamma2) * h_k

        return h_k, d_h, dd_h
    
    # ---- Backup CBF ----
    def get_backup_horizon(self):
        return float(getattr(self, "T_horizon", 2.0))

    def clamp_tau(self, tau):
        if tau is None:
            return None
        T = self.get_backup_horizon()
        tau_f = float(tau)
        return max(0.0, min(tau_f, T))

    def set_occ_barrier_fn(self, fn):
        self._occ_barrier_fn = fn

    def _occ_barrier(self, pos, scenario, tau=None):
        fn = getattr(self, "_occ_barrier_fn", None)
        if fn is None:
            raise AttributeError("Occlusion barrier function not set.")
        if not isinstance(scenario, dict):
            raise TypeError(f"scenario must be dict, got {type(scenario)}")
        
        if tau is None:
            tau = self.get_backup_horizon()
        else:
            tau = self.clamp_tau(tau)

        return fn(pos, scenario, tau)
    
    def _occ_safe_velocity_reference_rollout(self, X, scenarios, t):

        if (scenarios is None) or (len(scenarios) == 0):
            return np.zeros(2, dtype=float)
        
        p = np.array([float(X[0,0]), float(X[1,0])], dtype=float)
        v = np.array([float(X[2,0]), float(X[3,0])], dtype=float)

        a_lim = float(self.robot_spec.get('a_max', 1.0))
        R     = self.robot_spec['radius']

        A_stack =[]
        for sc in scenarios:
            A    = sc['A']
            b0   = sc['b0']
            vadv = sc['v_adv_max']

            # classify static or dynamic obstacles
            if 'v_expand_vec' in sc:
                v_expand = sc['v_expand_vec'] # Shape (K,)
            else:
                v_expand = np.full(len(b0), sc['v_adv_max'])

            tau = float(t) if t is not None else 0.0
            tau = self.clamp_tau(tau)

            delta = R + v_expand * tau
            h_vec = (A @ p) - b0 - delta

            active = (h_vec >= 0.0)  # outside wrt inflated poly
            if np.any(active):
                # calculate selected normal vectors
                avg_normal = A[active].mean(axis=0)

                # Calculate the magnitude normal vector
                norm = np.linalg.norm(avg_normal)

                if norm > 1e-9:
                    direction = avg_normal / norm

                    v_target = direction * vadv

                    A_stack.append(v_target)
            else:
                best_idx = np.argmax(h_vec)
                selected_normal = A[best_idx]
                norm = np.linalg.norm(selected_normal)
                if norm > 1e-9:
                    direction = selected_normal / norm
                    v_target = direction * vadv
                    A_stack.append(v_target)

        if len(A_stack) == 0:
            return np.zeros(2, dtype=float)
        
        A_all = np.vstack(A_stack)
        # v_ref = A_all.mean(axis=0)

        v_avg = A_all.mean(axis=0)
        # v_norm = np.linalg.norm(v_avg)
        
        # if v_norm > 1e-9:
        #     max_speed = np.max([np.linalg.norm(v) for v in A_stack])
        #     # avg_speed = np.mean([np.linalg.norm(v) for v in A_stack])
        #     v_ref = (v_avg / v_norm) * max_speed
        # else:
        #     v_ref = v_avg
        # # print(f"DEBUG: v_ref from occlusion backup: {v_ref}")
        
        return v_avg
        
    def backup_input(self, X, k_a=1.0):
        """
        Using stop() function as backup policy
        """
        return self.stop(X, k_a=k_a)

    
    def backup_input_occlusion(self, X, occlusion_scenarios, t=None,
                            k_d=1.0, k_occ=1.0):
        # print(f"DEBUG: backup_input_occlusion CALLED with t={t}")

        a_lim  = float(self.robot_spec.get('a_max', 1.0))
        v_max  = float(self.robot_spec.get('v_max', 1.0))
        Kp = float(self.pid_occ_gains.get("Kp", 1.0))

        v = np.array([float(X[2,0]), float(X[3,0])], dtype=float)
        v_ref = self._occ_safe_velocity_reference_rollout(X, occlusion_scenarios, t)
        
        e = v - v_ref
        u_unsat = -Kp * e - k_d * v
        u = np.clip(u_unsat, -a_lim, a_lim)

        # limite acceleration when near v_max
        eps = 1e-6
        for i in range(2):
            if v[i] >= (v_max - eps) and u[i] > 0.0: u[i] = 0.0
            if v[i] <= (-v_max + eps) and u[i] < 0.0: u[i] = 0.0

        return u.reshape(2,1)


    def f_cl(self, X, occlusion_scenarios=None, t=None):
        """
        System dynamics as using backup policy u_b (Closed-Loop)
        """
        if occlusion_scenarios:
            u_b = self.backup_input_occlusion(X, occlusion_scenarios, t)
        else:
            u_b = self.backup_input(X)

        return np.array([X[2,0], X[3,0], u_b[0,0], u_b[1,0]]).reshape(4,1)
    
    def _dvref_dp_fd(self, X, scenarios, t, eps=1e-3):
        p = X[0:2,0].astype(float)
        vref = self._occ_safe_velocity_reference_rollout(X, scenarios, t)
        J = np.zeros((2,2))
        for j in range(2):
            Xp = X.copy(); Xp[j,0] = p[j] + eps
            vr = self._occ_safe_velocity_reference_rollout(Xp, scenarios, t)
            J[:,j] = (vr - vref)/eps
        return J
    
    def F_cl(self, X, occlusion_scenarios=None, t=None):
        Kp = float(self.pid_occ_gains.get("Kp", 1.0))
        k_d = 1.0
        A = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], float)
        Bv = -(Kp + k_d) * np.eye(2)
        if occlusion_scenarios is not None and t is not None:
            Jp = self._dvref_dp_fd(X, occlusion_scenarios, t)
        else:
            Jp = np.zeros((2,2))
        lower_left = Kp * Jp
        F = np.block([[np.zeros((2,2)), np.eye(2)],
                    [lower_left,      Bv      ]])
        return F
        
    def set_terminal_backup_context(self, occlusion_scenario, T, kappa=None, rho_T=0.05):
        self._term_occ_scenario = occlusion_scenario
        self._term_T = float(T)
        if kappa is not None:
            self.kappa = kappa
        self._term_rho = float(rho_T)
        self._term_grad_cache = None
    
    def h_b_stop(self, X):

        p_T = X[0:2, 0].astype(float)

        scenario = getattr(self, "_term_occ_scenario", None)
        T = float(getattr(self, "_term_T", self.get_backup_horizon()))
        rho_T = float(getattr(self, "_term_rho", 0.05))

        if scenario is not None:
            # print("loop in h_b_stop")
            h_tilde, grad_pos, _ = self._occ_barrier(p_T.reshape(2, 1), scenario, tau=T)
            # print(f"h_tilde: {h_tilde} | rho_T: {rho_T}")
            self._term_grad_cache = (grad_pos.reshape(1,2) if grad_pos is not None else None)
            return float(h_tilde) - rho_T

    def grad_h_b_stop(self, X):
        gp = getattr(self, "_term_grad_cache", None)
        if gp is not None:
            # print("loop in grad_h_b_stop")
            if gp.shape == (1,2):
                # Expand to state dimension: [grad_pos, 0, 0]
                return np.hstack([gp, np.array([[0.0, 0.0]])])
            return gp
    
    # def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None):
    #     """
    #     Compute the future trajectory (phi_b) and sensitivity matrix (Phi_b, STM) by following the backup controller from the current state x0.
    #     """
    #     from scipy.integrate import solve_ivp

    #     if hasattr(self, "pid_occ"):
    #         self.pid_occ["t_prev"] = None
    #         self.pid_occ["v_prev"] = None
        
    #     def augmented_dynamics(t, y):
    #         x = y[0:4]
    #         Phi = y[4:].reshape((4, 4))
            
    #         x_col = x.reshape(-1, 1)
    #         x_dot = self.f_cl(x_col, occlusion_scenarios, t).flatten()
    #         Phi_dot = self.F_cl(x_col, occlusion_scenarios, t) @ Phi
            
    #         return np.concatenate([x_dot, Phi_dot.flatten()])

    #     y0 = np.concatenate([x0.flatten(), np.eye(4).flatten()])
    #     t_eval = np.arange(0.0, T + 1e-9, dt)
        
    #     sol = solve_ivp(
    #         augmented_dynamics,
    #         [0, T],
    #         y0,
    #         t_eval=t_eval,
    #         dense_output=True
    #     )
        
    #     backup_traj = sol.y[0:4, :].T       # (N,4)
    #     stm_traj = sol.y[4:, :].T.reshape(-1, 4, 4)
        
    #     return backup_traj, stm_traj, t_eval

    def jac_f_cl_fd(self, X, occlusion_scenarios=None, t=None, eps=1e-4):
        """
        Central-difference approximation of A = ∂f_cl/∂x at (X,t).
        X: (4,1)
        returns: (4,4)
        """
        X = np.asarray(X, dtype=float).reshape(4, 1)
        f0 = self.f_cl(X, occlusion_scenarios, t).reshape(4,)

        J = np.zeros((4, 4), dtype=float)

        # scale-aware eps (optional): helps when states are large/small
        # eps_j = eps * (1.0 + abs(X[j,0]))
        for j in range(4):
            ej = np.zeros((4, 1), dtype=float)
            ej[j, 0] = 1.0
            eps_j = eps * (1.0 + abs(float(X[j, 0])))

            Xp = X + eps_j * ej
            Xm = X - eps_j * ej

            fp = self.f_cl(Xp, occlusion_scenarios, t).reshape(4,)
            fm = self.f_cl(Xm, occlusion_scenarios, t).reshape(4,)

            J[:, j] = (fp - fm) / (2.0 * eps_j)

        return J
    
    # def _aug_rhs(self, t, y, occlusion_scenarios=None, eps_A=1e-4):
    #     """
    #     RHS for augmented state y = [x; vec(Phi)].
    #     y: (4 + 16,)
    #     returns ydot: (4 + 16,)
    #     """
    #     x = y[:4].reshape(4, 1)
    #     Phi = y[4:].reshape(4, 4)

    #     xdot = self.f_cl(x, occlusion_scenarios, t).reshape(4,)

    #     A = self.jac_f_cl_fd(x, occlusion_scenarios, t, eps=eps_A)  # (4,4)
    #     Phidot = (A @ Phi).reshape(-1)

    #     return np.concatenate([xdot, Phidot])

    def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None, eps_A=1e-4):
        x = np.asarray(x0, float).reshape(4,1)
        Phi = np.eye(4)
        N = int(np.floor(T/dt)) + 1
        t_grid = dt*np.arange(N)

        backup_traj = np.zeros((N,4))
        stm_traj = np.zeros((N,4,4))
        backup_traj[0] = x.ravel()
        stm_traj[0] = Phi
        fcl_traj = np.zeros((N,4))

        I = np.eye(4)

        for k in range(N-1):
            t = float(t_grid[k])

            # 1) RK4 for x (NO Jacobian inside)
            k1 = self.f_cl(x, occlusion_scenarios, t)
            fcl_traj[k] = k1.ravel()
            k2 = self.f_cl(x + 0.5*dt*k1, occlusion_scenarios, t+0.5*dt)
            k3 = self.f_cl(x + 0.5*dt*k2, occlusion_scenarios, t+0.5*dt)
            k4 = self.f_cl(x + dt*k3,      occlusion_scenarios, t+dt)
            x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

            # 2) A once per step (use midpoint)
            x_mid = x + 0.5*dt*k2
            A = self.jac_f_cl_fd(x_mid, occlusion_scenarios, t+0.5*dt, eps=eps_A)

            # 3) Phi update (cheap)
            Phi_next = (I + dt*A) @ Phi

            x, Phi = x_next, Phi_next
            backup_traj[k+1] = x.ravel()
            stm_traj[k+1] = Phi
        
        fcl_traj[-1] = self.f_cl(x, occlusion_scenarios, float(t_grid[-1])).ravel()

        return backup_traj, stm_traj, t_grid, fcl_traj

    # def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None, eps_A=1e-4):
    #     """
    #     Fixed-step RK4 rollout + FD-Jacobian STM propagation.

    #     Returns:
    #     backup_traj: (N,4)
    #     stm_traj:    (N,4,4)
    #     t_grid:      (N,)
    #     """
    #     x0 = np.asarray(x0, dtype=float).reshape(4, 1)

    #     # fixed time grid
    #     N = int(np.floor(T / dt)) + 1
    #     t_grid = dt * np.arange(N, dtype=float)
    #     # ensure final time hits T (optional)
    #     if abs(t_grid[-1] - T) > 1e-12:
    #         t_grid = np.append(t_grid, T)
    #         N = len(t_grid)

    #     # initial augmented state
    #     Phi0 = np.eye(4, dtype=float)
    #     y = np.concatenate([x0.reshape(-1), Phi0.reshape(-1)])

    #     backup_traj = np.zeros((N, 4), dtype=float)
    #     stm_traj = np.zeros((N, 4, 4), dtype=float)

    #     backup_traj[0, :] = x0.reshape(-1)
    #     stm_traj[0, :, :] = Phi0

    #     for k in range(N - 1):
    #         t = float(t_grid[k])
    #         h = float(t_grid[k + 1] - t_grid[k])

    #         # RK4 on augmented system
    #         k1 = self._aug_rhs(t,         y,               occlusion_scenarios, eps_A)
    #         k2 = self._aug_rhs(t + 0.5*h, y + 0.5*h*k1,    occlusion_scenarios, eps_A)
    #         k3 = self._aug_rhs(t + 0.5*h, y + 0.5*h*k2,    occlusion_scenarios, eps_A)
    #         k4 = self._aug_rhs(t + h,     y + h*k3,        occlusion_scenarios, eps_A)

    #         y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    #         xk = y[:4]
    #         Phik = y[4:].reshape(4, 4)

    #         backup_traj[k + 1, :] = xk
    #         stm_traj[k + 1, :, :] = Phik

    #     return backup_traj, stm_traj, t_grid
