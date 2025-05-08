import numpy as np
import casadi as ca

"""
Created on July 11th, 2024
@author: Taekyung Kim

@description: 
Dynamic unicycle model for CBF-QP and MPC-CBF (casadi)
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


class DynamicUnicycle2D_C3BF:

    def __init__(self, dt, robot_spec):
        """
        '''Continuous Time C3BF'''
            Compute a Collision Cone Control Barrier Function for the Dynamic Unicycle2D.
            X: [x, y, theta, v, ω]
            U: [a, alpha]
            The barrier's relative degree is "1"
                h_dot = ∂h/∂x ⋅ f(x) + ∂h/∂x ⋅ g(x) ⋅ u

            Define h from the collision cone idea:
                p_rel = [obs_x - (x + l * cos(theta)), obs_y - (y + l * sin(theta)]
                v_rel = [obs_x_dot - (v * cos(theta) - l * sin(theta) * ω), obs_y_dot - (v * sin(theta) + l * cos(theta) * ω)]
                dist = ||p_rel||
                R = ro
        """

        self.dt = dt
        self.robot_spec = robot_spec

        if 'a_max' not in self.robot_spec:
            self.robot_spec['a_max'] = 0.5
        if 'w_max' not in self.robot_spec:
            self.robot_spec['w_max'] = 1.0
        if 'alpha_max' not in self.robot_spec:
            self.robot_spec['alpha_max'] = 0.5
        if 'v_max' not in self.robot_spec:
            self.robot_spec['v_max'] = 1.0
        if 'rear_ax_distance' not in self.robot_spec:
            self.robot_spec['rear_ax_dist'] = 0.2

    def f(self, X, casadi=False):
        if casadi:
            return ca.vertcat(
                X[3, 0] * ca.cos(X[2, 0]),
                X[3, 0] * ca.sin(X[2, 0]),
                X[4, 0],
                0,
                0
            )
        else:
            return np.array([X[3, 0]*np.cos(X[2, 0]),
                            X[3, 0]*np.sin(X[2, 0]),
                            X[4, 0],
                            0,
                            0]).reshape(-1, 1)

    def df_dx(self, X):
        return np.array([
            [0, 0, -X[3, 0]*np.sin(X[2, 0]), np.cos(X[2, 0])],
            [0, 0,  X[3, 0]*np.cos(X[2, 0]), np.sin(X[2, 0])],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def g(self, X, casadi=False):
        if casadi:
            return ca.DM([
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
            ])
        else:
            return np.array([[0, 0], [0, 0], [0, 0], [1, 0], [0, 1] ])

    def step(self, X, U):
        X = X + (self.f(X) + self.g(X) @ U)*self.dt
        X[2, 0] = angle_normalize(X[2, 0])
        return X

    def nominal_input(self, X, G, d_min=0.05, k_omega=2.0, k_a=1.0, k_v=1.0):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(G.reshape(-1, 1))  # goal state
        v_max = self.robot_spec['v_max']
        alpha_max = self.robot_spec['alpha_max']

        # don't need a min dist since it has accel
        distance = max(np.linalg.norm(X[0:2, 0]-G[0:2, 0]) - d_min, 0.0)
        theta_d = np.arctan2(G[1, 0]-X[1, 0], G[0, 0]-X[0, 0])
        error_theta = angle_normalize(theta_d - X[2, 0])

        omega = k_omega * error_theta
        k_alpha = 3.0
        alpha = np.clip(k_omega*(omega - X[4, 0]), -alpha_max, alpha_max) # derivative of omega = alpha

        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = min(k_v * distance * np.cos(error_theta), v_max)
        # print("distance: ", distance, "v: ", v, "error_theta: ", error_theta)

        accel = k_a * (v - X[3, 0])
        # print(f"CBF nominal acc: {accel}, alpha:{alpha}")
        return np.array([accel, alpha]).reshape(-1, 1)

    def stop(self, X, k_a=1.0):
        # set desired velocity to zero
        v = 0.0
        accel = k_a * (v - X[3, 0])
        return np.array([accel, 0]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        return np.linalg.norm(X[3, 0]) < tol
    
    def rotate_to(self, X, theta_des, k_omega=2.0):
        error_theta = angle_normalize(theta_des - X[2, 0])
        omega = k_omega * error_theta
        alpha = k_omega * (omega - X[4, 0])
        return np.array([0.0, alpha]).reshape(-1, 1)

    def agent_barrier(self, X, obs, robot_radius, beta=1.0):

        theta = X[2, 0]
        v = X[3, 0]
        omega = X[4, 0]
        L_r = self.robot_spec['rear_ax_dist']
        print(f"theta: {theta} | v: {v} | omega: {omega}")
        
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
        p_rel = np.array([[obs[0] - (X[0, 0] + L_r * np.cos(theta))], 
                        [obs[1] - (X[1, 0] + L_r * np.sin(theta))]])
        v_rel = np.array([[obs_vel_x - (v * np.cos(theta) - L_r * np.sin(theta) * omega)], 
                        [obs_vel_y - (v * np.sin(theta) + L_r * np.cos(theta) * omega)]])

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
        dh_dx = np.zeros((1, 5))
        dh_dx[0, 0] = -v_rel_x - v_rel_mag * p_rel_x / d_x
        dh_dx[0, 1] = -v_rel_y - v_rel_mag * p_rel_y / d_x
        dh_dx[0, 2] =  L_r * np.sin(theta) * v_rel_x + p_rel_x * (v * np.sin(theta) + L_r * np.cos(theta) * omega) - L_r * np.cos(theta) * v_rel_y + p_rel_y * (-v * np.cos(theta) + L_r * np.sin(theta) * omega) + d_x / v_rel_mag * (v_rel_x * (v* np.sin(theta)+L_r*np.cos(theta)*omega) + v_rel_y * (-v*np.cos(theta)+L_r*np.sin(theta)*omega)) + v_rel_mag / d_x * (p_rel_x * L_r * np.sin(theta) - p_rel_y * L_r * np.cos(theta))
        dh_dx[0, 3] = - p_rel_x * np.cos(theta) - p_rel_y * np.sin(theta)  - (d_x / v_rel_mag * (v_rel_x * np.cos(theta) + v_rel_y * np.sin(theta)))
        dh_dx[0, 4] = L_r * d_x / v_rel_mag * (v_rel_x * np.sin(theta) - v_rel_y * np.cos(theta))

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
        