from robots.dynamic_unicycle2D import DynamicUnicycle2D
import numpy as np
import casadi as ca

"""
Created on July 11th, 2024
@author: Taekyung Kim

@description: 
Dynamic unicycle model for CBF-QP and MPC-CBF (casadi)
"""

class DynamicUnicycle2D_C3BF(DynamicUnicycle2D):
    def __init__(self, dt, robot_spec):
        super().__init__(dt, robot_spec)

    def agent_barrier(self, X, obs, robot_radius, beta=1.0):

        theta = X[2, 0]
        v = X[3, 0]
        
        # Check if obstacles have velocity components (static or moving)
        if obs.shape[0] > 3:
            obs_vel_x = obs[3]
            obs_vel_y = obs[4]

        else:
            obs_vel_x = 0.0
            obs_vel_y = 0.0

        # Combine radius R
        ego_dim = (obs[2] + robot_radius) * beta # Total collision safe radius

        # Compute relative position and velocity
        p_rel = np.array([[obs[0] - X[0, 0]], 
                        [obs[1] - X[1, 0]]])
        v_rel = np.array([[obs_vel_x - (v * np.cos(theta))], 
                        [obs_vel_y - (v * np.sin(theta))]])

        p_rel_x = p_rel[0, 0]
        p_rel_y = p_rel[1, 0]
        v_rel_x = v_rel[0, 0]
        v_rel_y = v_rel[1, 0]

        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)

        # Compute cos_phi safely for c3bf
        eps = 1e-6
        dist = np.maximum(p_rel_mag**2 - ego_dim**2, eps)
        d_x = np.sqrt(dist)
        cos_phi = d_x / p_rel_mag
        
        # Compute h
        h = np.dot(p_rel.T, v_rel)[0, 0] + p_rel_mag * v_rel_mag * cos_phi

        # Compute dh_dx
        dh_dx = np.zeros((1, 4))
        dh_dx[0, 0] = -v_rel_x - v_rel_mag * p_rel_x / d_x
        dh_dx[0, 1] = -v_rel_y - v_rel_mag * p_rel_y / d_x
        dh_dx[0, 2] = p_rel_x * v * np.sin(theta) + p_rel_y * (-v * np.cos(theta)) + d_x / v_rel_mag * (v_rel_x * v* np.sin(theta) + v_rel_y * (-v*np.cos(theta)))
        dh_dx[0, 3] = - p_rel_x * np.cos(theta) - p_rel_y * np.sin(theta)  - (d_x / v_rel_mag * (v_rel_x * np.cos(theta) + v_rel_y * np.sin(theta)))
        return h, dh_dx

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
        

    # def render_rigid_body(self, X, U):
    #     """
    #     5‑state Dynamic Unicycle 의 rigid body + two wheels 렌더링용 transform 반환.
    #     X = [x, y, theta, v, omega]
    #     U = [a, alpha]  (여기서는 omega 에만 영향; 시각화엔 쓰이지 않음)
    #     """
    #     # 1) 상태 추출
    #     x, y, theta, v, omega = X.flatten()

    #     # 2) body / wheel 크기 (robot_spec 으로 조정 가능)
    #     L_body = self.robot_spec.get('body_length', 0.6)    # 전체 길이
    #     W_body = self.robot_spec.get('body_width',  0.3)    # 전체 폭
    #     Lr = self.robot_spec['rear_ax_dist']
    #     Lf = self.robot_spec['front_ax_dist']

    #     x_p = x + Lr * np.cos(theta)
    #     y_p = y + Lr * np.sin(theta)
    #     # 3) Affine 변환 생성
    #     transform_body = Affine2D().rotate(theta).translate(x_p, y_p) + plt.gca().transData

    #     # 4) 축 위치 계산 (중심으로부터 앞/뒤 축 거리)

    #     # 5) body, rear_wheel, front_wheel patch 생성 (한 번만 __init__ 에서)
    #     #    → 여기서는 transform 만 돌려줍니다.

    #     # body (Rectangle)
    #     transform_rear = (Affine2D()
    #                         .rotate(theta)
    #                         .translate(x, y) + plt.gca().transData)
    #     # rear wheel (Circle)
    #     #   body 로컬 좌표계에서 뒤축은 (-Lr, 0)

    #     # front wheel (Circle)
    #     #   body 로컬 좌표계에서 앞축은 (+Lf, 0)
    #     x_f = x_p + Lf * np.cos(theta)
    #     y_f = y_p + Lf * np.sin(theta)
    #     transform_front = (Affine2D()
    #                         .rotate(theta)
    #                         .translate(x_f, y_f) + plt.gca().transData)
    #     # transform_front = transform_body + front_offset

    #     return transform_body, transform_body, transform_body