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
            "Kp": 1.0,    # 비례 (속도 오차)
            "Ki": 0.2,    # 적분
            "Kd": 0.1,    # 미분
            "aw_limit": 2.0 * self.robot_spec.get("a_max", 1.0)  # anti-windup 한계
        }
        self.occ_margin = 1.0
        # look-ahead 샘플 (백업 입력 설계용 가벼운 프리뷰)
        self.occ_tau_preview = [0.0, 0.3, 0.6]

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
    
    #### Backup CBF ####
    def set_occ_barrier_fn(self, fn):
        """
        외부(백업 컨트롤러)에서 curved occlusion barrier 함수를 주입함.
        기대 시그니처: fn(pos:(2,1) or (2,), scenario:dict, tau:float)
        -> (h_tilde:float or None, grad_pos:(1,2) or None, risk_vec: (M,2) or None)
        """
        self._occ_barrier_fn = fn

    def _occ_barrier(self, pos, scenario, tau=0.0):
        fn = getattr(self, "_occ_barrier_fn", None)
        if fn is None:
            raise AttributeError("Occlusion barrier function not set.")
        if not isinstance(scenario, dict):
            raise TypeError(f"scenario must be dict, got {type(scenario)}")
        return fn(pos, scenario, tau)
    
    def _occ_safe_velocity_reference_single(self, X, scenario):
        """
        단일 장애물 시나리오에 대해서만 v_ref 생성:
        tau 프리뷰마다 a=max(0, rho - h_tilde)로 가중합한 (-grad) 방향.
        """
        p = X[0:2, 0].astype(float)
        v = X[2:4, 0].astype(float)
        rho = float(self.occ_margin)
        eps = 1e-9

        v_ref = np.zeros(2)
        for tau in self.occ_tau_preview:
            p_tau = p + tau * v
            h_tilde, grad_pos, _ = self._occ_barrier(p_tau.reshape(2,1), scenario, tau=tau)
            if (h_tilde is None) or (grad_pos is None):
                continue
            a = max(0.0, rho - float(h_tilde))
            if a <= 0.0:
                continue
            n = grad_pos.reshape(2,)
            nrm = np.linalg.norm(n)
            if nrm < eps:
                continue
            v_ref += a * ( - n / nrm )  # 도피: -grad

        # 크기 제한
        vref_max = float(self.pid_occ_gains.get("vref_max", 0.5))
        m = np.linalg.norm(v_ref)
        if m > vref_max and m > 0.0:
            v_ref = v_ref * (vref_max / m)
        return v_ref

    
    def _occ_safe_velocity_reference(self, X, occlusion_scenarios, k_occ=1.0):
        """
        여러 시나리오와 소수의 look-ahead tau에서 h_tilde<rho 인 곳의
        기울기(-grad)를 가중합하여 '도피' 목표 속도 v_ref를 만든다.
        """
        p = X[0:2, 0].astype(float)
        v = X[2:4, 0].astype(float)
        if not occlusion_scenarios:
            return np.zeros(2)

        rho = float(self.occ_margin)
        v_ref = np.zeros(2)
        eps = 1e-9

        # 가벼운 위치 프리뷰: x_tau ≈ p + tau * v (constant velocity 근사)
        A_stack = []
        tan_risk = occlusion_scenarios.get('risk_normal_vec', None)
        arc_risk = occlusion_scenarios.get('arc_adv', None)
                # print(f"tan_risk: {tan_risk} || arc_risk: {arc_risk}")
                #A_stack.append(tan_risk)
        A_stack.append(arc_risk)
        # print(f"A_stack: {A_stack}")
        if len(A_stack) > 0:
            A_all = np.vstack(A_stack)  # (M_tot, 2)
            # soft aggrgation: use the average normal (signs are already chosen so that they point toward the unsafe side)
            v_ref = A_all.mean(axis=0)
            # print(f"A_all: {A_all} || v_ref: {v_ref}")
                # n_norm = np.linalg.norm(n)
                # v_dot_n = float(v @ n_mean)
                # if v_dot_n > 0.0:
                #         u -= k_occ * v_dot_n * n_mean
                        
                # if n_norm > 1e-6:
                #     n = n / n_norm  # normalized unsafe-direction normal
                #     v_dot_n = float(v @ n)
                #     # If v_dot_n <0 means along the unsafe direction
                #     if v_dot_n > 0.0:
                #         u -= k_occ * v_dot_n * n

        # saturation
        # a_max = float(self.robot_spec.get('a_max', 1.0))
        # norm_u = np.linalg.norm(u)
        # if norm_u > a_max and norm_u > 0.0:
        #     u = u * (a_max / norm_u)
        # a_lim = float(self.robot_spec.get('a_max', 1.0))
        # u = np.clip(u, -a_lim, a_lim)
        # print(f"u: {u}")

        return v_ref
    
    def backup_input(self, X, k_a=1.0):
        """
        Using stop() function as backup policy
        """
        return self.stop(X, k_a=k_a)
    
    #### Backup policy for occlusion-aware adversaries #########
    # def backup_input_occlusion(self, X, occlusion_scenarios, k_d=1.0, k_occ=1.0):
    #     """
    #     Occlusion-aware backup policy:
    #     - By default, apply velocity damping.
    #     - If there is a velocity component towards an occluded-unsafe direction, push the velocity away from that direction.
    #     """
    #     v = X[2:4, 0].astype(float)
    #     # u = np.zeros(2)
    #     u = -k_d * v  # base damping

    #     # if occlusion_scenarios:
    #     #     # Collect all facet normals from every occlusion scenario
    #     #     A_stack = []
    #     #     for sc in occlusion_scenarios:
    #     #         A = sc.get('A', None)
    #     #         if A is not None and A.size > 0:
    #     #             A_stack.append(A)
    #     #     if len(A_stack) > 0:
    #     #         A_all = np.vstack(A_stack)  # (M_tot, 2)
    #     #         # soft aggrgation: use the average normal (signs are already chosen so that they point toward the unsafe side)
    #     #         n = -A_all.mean(axis=0)
    #     #         n_norm = np.linalg.norm(n)
    #     #         if n_norm > 1e-6:
    #     #             n = n / n_norm  # normalized unsafe-direction normal
    #     #             v_dot_n = float(v @ n)
    #     #             # If v_dot_n <0 means along the unsafe direction
    #     #             if v_dot_n > 0.0:
    #     #                 u -= k_occ * v_dot_n * n
        
    #     if occlusion_scenarios:
    #         # Collect all facet normals from every occlusion scenario
    #         A_stack = []
    #         for sc in occlusion_scenarios:
    #             tan_risk = sc.get('risk_normal_vec', None)
    #             arc_risk = sc.get('arc_adv', None)
    #             # print(f"tan_risk: {tan_risk} || arc_risk: {arc_risk}")
    #             if tan_risk is not None and tan_risk.size > 0:
    #                 A_stack.append(tan_risk)
    #                 A_stack.append(arc_risk)
    #         if len(A_stack) > 0:
    #             A_all = np.vstack(A_stack)  # (M_tot, 2)
    #             # soft aggrgation: use the average normal (signs are already chosen so that they point toward the unsafe side)
    #             n_mean = A_all.mean(axis=0)
    #             # print(f"A_all: {A_all} || n_mean: {n_mean}")
    #             # n_norm = np.linalg.norm(n)
    #             v_dot_n = float(v @ n_mean)
    #             if v_dot_n > 0.0:
    #                     u -= k_occ * v_dot_n * n_mean
                        
    #             # if n_norm > 1e-6:
    #             #     n = n / n_norm  # normalized unsafe-direction normal
    #             #     v_dot_n = float(v @ n)
    #             #     # If v_dot_n <0 means along the unsafe direction
    #             #     if v_dot_n > 0.0:
    #             #         u -= k_occ * v_dot_n * n

    #     # saturation
    #     # a_max = float(self.robot_spec.get('a_max', 1.0))
    #     # norm_u = np.linalg.norm(u)
    #     # if norm_u > a_max and norm_u > 0.0:
    #     #     u = u * (a_max / norm_u)
    #     a_lim = float(self.robot_spec.get('a_max', 1.0))
    #     u = np.clip(u, -a_lim, a_lim)
    #     # print(f"u: {u}")
                    
    #     return u.reshape(2, 1)
    
    def backup_input_occlusion(self, X, occlusion_scenarios, k_d=1.0, k_occ=1.0):
        """
        PID 기반 백업 정책:
        1) occlusion 위험을 보고 도피 목표속도 v_ref 생성
        2) 현재 속도 v를 v_ref로 추종하는 PID 설계 (출력=가속도 u)
        3) 기본 감쇠 -k_d v 를 더해 안정화
        """
        dt = float(self.dt)
        a_lim = float(self.robot_spec.get('a_max', 1.0))

        # v = X[2:4, 0].astype(float)
        v_x = float(X[2, 0])
        v_y = float(X[3, 0])
        v = np.array([v_x, v_y], dtype=float)
        # print(f"current vel: {v} | v_x: {v_x} | v_y: {v_y}")

        # 1) 위험 기반 목표 속도
        v_ref = self._occ_safe_velocity_reference(X, occlusion_scenarios[0])  # (2,)
        
        # 2) PID on velocity error
        gains = self.pid_occ_gains
        e = v - v_ref
        # print(f"v: {v} | v_ref: {v_ref} | e: {e}")
        self.pid_occ["I"] += e * dt
        D = (e - self.pid_occ["e_prev"]) / max(dt, 1e-6)
        self.pid_occ["e_prev"] = e

        u_pid = -gains["Kp"] * e - gains["Ki"] * self.pid_occ["I"] - gains["Kd"] * D

        # Anti-windup(단순 클램핑)
        aw = gains["aw_limit"]
        self.pid_occ["I"] = np.clip(self.pid_occ["I"], -aw, aw)

        # 3) 기본 전역 감쇠(브레이크)
        u = u_pid - k_d * v

        # 포화
        u = np.clip(u, -a_lim, a_lim)
        # print(f"u: {u}")
        return u.reshape(2,1)

    def f_cl(self, X, occlusion_scenarios=None):
        """
        System dynamics as using backup policy u_b (Closed-Loop)
        """
        if occlusion_scenarios:
            u_b = self.backup_input_occlusion(X, occlusion_scenarios)
        else:
            u_b = self.backup_input(X)
            
        # dt = float(self.dt)
        # v_max = float(self.robot_spec.get('v_max', np.inf))
        # if np.isfinite(v_max):
        #     v_now = X[2:4, 0].astype(float)           # (2,)
        #     v_pred = v_now + (u_b.reshape(2,) * dt)   # 오일러 예측
        #     n = np.linalg.norm(v_pred)
        #     if n > v_max and n > 0.0:
        #         v_pred = v_pred * (v_max / n)
        #         u_b = ((v_pred - v_now) / dt).reshape(2,1)
                
        return self.f(X) + self.g(X) @ u_b
    
        # u_b = self.backup_input(X)
        # return self.f(X) + self.g(X) @ u_b

    def F_cl(self, X):
        """
        Jacobian matrix of f_cl(X)
        Use for STM calculation
        As u_b = k_a * (-v), df_cl/dv = -k_a
        """
        k_a = 1.0
        return np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, -k_a, 0],
            [0, 0, 0, -k_a]
        ])
        
    def set_terminal_backup_context(self, occlusion_scenario, T, kappa=None, rho_T=0.5):
        # self._term_occ_scenarios = occlusion_scenarios if occlusion_scenarios else []
        self._term_occ_scenario = occlusion_scenario
        self._term_T = float(T)
        if kappa is not None:
            self.kappa = kappa
        self._term_rho = float(rho_T)
        self._term_grad_cache = None
    
    def h_b_stop(self, X):
        """
        터미널 백업 집합 조건:
        h_b = min_s h_tilde_curved(p_T; tau=T) - rho_T
        (시나리오가 없으면 기존 속도 기반 안전셋으로 폴백)
        """
        p_T = X[0:2, 0].astype(float)

        # scenarios = getattr(self, "_term_occ_scenarios", [])
        scenario = getattr(self, "_term_occ_scenario", None)
        T = float(getattr(self, "_term_T", 3.0))
        rho_T = float(getattr(self, "_term_rho", 0.5))
        
        best_h = np.inf
        best_grad = None

        # if scenarios:
        #     h_vals = []
        #     for sc in scenarios:
        #         h_tilde, _, _ = self._occ_barrier(
        #             p_T.reshape(2,1), sc, tau=T
        #         )
        #         print(f"h_tilde: {h_tilde}")
        #         if h_tilde is not None and np.isfinite(h_tilde):
        #             h_vals.append(float(h_tilde))
        #     if len(h_vals) > 0:
        #         # U_T로부터의 여유: h_tilde - rho_T
        #         return np.min(h_vals) - rho_T
        if scenario is not None:
            h_tilde, grad_pos, _ = self._occ_barrier(p_T.reshape(2, 1), scenario, tau=T)
            # print(f"h_tilde: {h_tilde}")
            self._term_grad_cache = (grad_pos.reshape(1,2) if grad_pos is not None else None)
            return float(h_tilde) - rho_T

        # # # 폴백: 예전 속도 기반(느리게 멈춘 상태)
        # v_sq = X[2, 0]**2 + X[3, 0]**2
        # v_safe_sq = (0.3)**2
        # return v_safe_sq - v_sq

    def grad_h_b_stop(self, X):
        """
        h_b_stop의 p-그라디언트:
        argmin 시나리오 s*의 grad_pos(p_T; tau=T)를 사용.
        전체 상태 그라디언트는 [grad_pos, 0, 0].
        (시나리오 없으면 속도 기반 폴백의 기울기)
        """
        # p_T = X[0:2, 0].astype(float)

        # # scenarios = getattr(self, "_term_occ_scenarios", [])
        # scenario = getattr(self, "_term_occ_scenario", None)
        # T = float(getattr(self, "_term_T", 0.0))

        # if scenario:
        #     print("loop in")
        #     h_tilde, grad_pos, _ = self._occ_barrier(p_T.reshape(2, 1), scenario, tau=T)
        #     # if grad_pos:
        #     #     # (1,2) -> [grad_pos, 0, 0]
        #     return np.hstack([grad_pos.reshape(1, 2), np.array([[0.0, 0.0]])])
            
        gp = getattr(self, "_term_grad_cache", None)
        if gp is not None:
            if gp.shape == (1,2):
                # 확장: [grad_pos, 0, 0]
                return np.hstack([gp, np.array([[0.0, 0.0]])])
            # 이미 (1,4) (속도 기반 폴백에서 설정)
            return gp
        
        # if scenarios:
        #     # 최악 시나리오 선택
        #     best_h = np.inf
        #     best_grad = None
        #     for sc in scenarios:
        #         h_tilde, grad_pos, _ = self._occ_barrier(
        #             p_T.reshape(2,1), sc, tau=T
        #         )
        #         if (h_tilde is not None) and (grad_pos is not None) and np.isfinite(h_tilde):
        #             if h_tilde < best_h:
        #                 best_h = float(h_tilde)
        #                 best_grad = grad_pos.reshape(1,2)

        #     if best_grad is not None:
        #         # 전체 상태로 확장: [grad_pos, 0, 0]
        #         return np.hstack([best_grad, np.array([[0.0, 0.0]])])

        # 폴백: 속도 기반
        #return np.array([[0, 0, -2 * X[2, 0], -2 * X[3, 0]]])
    
    ## Min vel terminal set
    # def h_b_stop(self, X):
    #     """
    #     Define Backup Set h_b(x) >= 0. (S_0)
    #     h_b = v_max^2 - (vx^2 + vy^2) >= 0
    #     """
    #     v_sq = X[2, 0]**2 + X[3, 0]**2
    #     v_safe_sq = (0.3)**2
    #     return v_safe_sq - v_sq

    # def grad_h_b_stop(self, X):
    #     """
    #     Gradient of h_b_stop
    #     """
    #     return np.array([[0, 0, -2 * X[2, 0], -2 * X[3, 0]]])
    
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
        t_eval = np.arange(0, T, dt)
        
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