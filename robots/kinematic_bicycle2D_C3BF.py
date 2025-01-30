import numpy as np
import sympy
import casadi as ca

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

"""
Simplified kinematic bicycle model for CBF-QP and MPC-CBF (casadi)
https://arxiv.org/abs/2403.07043
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


class KinematicBicycle2D_C3BF:
    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, theta, v]
            U: [a, β] (acceleration, slip angle as control input)
            c3bf: h(x) = <p_rel, v_rel> + ||p_rel|| * ||v_rel|| * cos(phi)
            
            Equations:
            β = arctan((L_r / L) * tan(δ)) (Slip angle)
            
            x_dot = v * cos(theta) - v * sin(theta) * β
            y_dot = v * sin(theta) + v * cos(theta) * β
            theta_dot = (v / L_r) * β
            v_dot = a
        '''
        self.dt = dt
        self.robot_spec = robot_spec

        if 'wheel_base' not in self.robot_spec:
            self.robot_spec['wheel_base'] = 0.5
        if 'body_width' not in self.robot_spec:
            self.robot_spec['body_width'] = 0.3
        if 'radius' not in self.robot_spec:
            self.robot_spec['radius'] = 0.5
        if 'front_ax_dist' not in self.robot_spec:
            self.robot_spec['front_ax_dist'] = 0.2
        if 'rear_ax_distance' not in self.robot_spec:
            self.robot_spec['rear_ax_dist'] = 0.3
        if 'v_max' not in self.robot_spec:
            self.robot_spec['v_max'] = 1.0
        if 'a_max' not in self.robot_spec:
            self.robot_spec['a_max'] = 0.5
        if 'delta_max' not in self.robot_spec:
            self.robot_spec['delta_max'] = np.deg2rad(30)
        if 'beta_max' not in self.robot_spec:
            self.robot_spec['beta_max'] = self.beta(self.robot_spec['delta_max'])

    def beta(self, delta):
        # Computes the slip angle beta
        L_r = self.robot_spec['rear_ax_dist']
        L = self.robot_spec['wheel_base']
        return np.arctan((L_r / L) * np.tan(delta))
            
    def beta_to_delta(self, beta):
        # Map slip angle beta to steering angle delta
        L_r = self.robot_spec['rear_ax_dist']
        L = self.robot_spec['wheel_base']
        return np.arctan((L / L_r) * np.tan(beta))

            
    def df_dx(self, X):
        return np.array([
            [0, 0, -X[3, 0]*np.sin(X[2, 0]), np.cos(X[2, 0])],
            [0, 0,  X[3, 0]*np.cos(X[2, 0]), np.sin(X[2, 0])],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def f(self, X, casadi=False):
        theta = X[2, 0]
        v = X[3, 0]
        if casadi:
            return ca.vertcat(
                v * ca.cos(theta), 
                v * ca.sin(theta), 
                0,                 
                0                  
            )
        else:
            return np.array([
                v * np.cos(theta), 
                v * np.sin(theta), 
                0,                 
                0                  
            ]).reshape(-1, 1)

    def g(self, X, casadi=False):
        theta = X[2, 0]
        v = X[3, 0]
        L_r = self.robot_spec['rear_ax_dist']
        if casadi:
            g = ca.SX.zeros(4, 2)
            g[0, 1] = -v * ca.sin(theta) 
            g[1, 1] = v * ca.cos(theta)  
            g[2, 1] = v / L_r       
            g[3, 0] = 1                  
            return g
        else:
            return np.array([
                [0, -v * np.sin(theta)],
                [0, v * np.cos(theta)],
                [0, v / L_r],
                [1, 0]
            ])

    def step(self, X, U):
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        X[2, 0] = angle_normalize(X[2, 0])
        return X
   
    def nominal_input(self, X, G, d_min=0.05, k_theta=1.0, k_a=1.5, k_v=0.5):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(G.reshape(-1, 1))  # goal state
        v_max = self.robot_spec['v_max']
        delta_max = self.robot_spec['delta_max']

        distance = max(np.linalg.norm(X[0:2, 0] - G[0:2, 0]) - d_min, 0.05)
        theta_d = np.arctan2(G[1, 0] - X[1, 0], G[0, 0] - X[0, 0])
        error_theta = angle_normalize(theta_d - X[2, 0])

        # Steering angle and slip angle
        delta = np.clip(k_theta * error_theta, -delta_max, delta_max)  # Steering angle
        beta = self.beta(delta) # Slip angle conversion
                
        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = min(k_v * distance * np.cos(error_theta), v_max)
            
        a = k_a * (v - X[3, 0])

        print(f"a: {a}, beta: {beta}")
        return np.array([a, beta]).reshape(-1, 1)
    
    def stop(self, X):
        return np.array([0, 0]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        return abs(X[3, 0]) < tol  

    def rotate_to(self, X, theta_des, k_theta=2.0):
        error_theta = angle_normalize(theta_des - X[2, 0])
        beta = k_theta * error_theta
        return np.array([0.0, beta]).reshape(-1, 1)
    
    def render_rigid_body(self, X, U):
        '''
        Return the materials to render the rigid body
        '''
        x, y, theta, v = X.flatten()
        beta = U[1, 0]  # Steering angle control input
        delta = self.beta_to_delta(beta)

        # Update vehicle body
        transform_body = Affine2D().rotate(theta).translate(x, y) + plt.gca().transData

        # Calculate axle positions
        rear_axle_x = x - self.robot_spec['rear_ax_dist'] * np.cos(theta)
        rear_axle_y = y - self.robot_spec['rear_ax_dist'] * np.sin(theta)
        front_axle_x = x + self.robot_spec['front_ax_dist'] * np.cos(theta)
        front_axle_y = y + self.robot_spec['front_ax_dist'] * np.sin(theta)

        # Update rear wheel (aligned with vehicle orientation)
        transform_rear = (Affine2D()
                            .rotate(theta)
                            .translate(rear_axle_x, rear_axle_y) + plt.gca().transData)

        # Update front wheel (rotated by steering angle)
        transform_front = (Affine2D() 
                            .rotate(theta + delta)
                            .translate(front_axle_x, front_axle_y) + plt.gca().transData)
    
        return transform_body, transform_rear, transform_front
    
    def agent_barrier(self, X, obs, robot_radius, beta=1.0):
        """
        Compute a Collision Cone Control Barrier Function for the Kinematic Bicycle (continous time).
        
        X: [x, y, theta, v]
        return: (h, h_dot, dh_dot_dx)
            h           : scalar, the C3BF value
            dh_dx       : (1, 4) array, gradient of h

        The barrier is of "relative degree of 1" if done in velocity space, but for a kinematic bicycle, we can follow a pattern similar to agent_barrier:
            h_dot = ∂h/∂x ⋅ f(x) + ∂h_dot/∂x ⋅ g(x) ⋅ u

        Define h from the collision cone idea:
            p_rel = [obs_x - x, obs_y - y]
            v_rel = [-v_cos(theta), -v_sin(theta)] (since obstacle is static)
            dist = ||p_rel||
            R = robot_radius + obs_r (combined)
            inside = dist^2 - R^2

            if inside <= 0 -> already in collision possibility -> big negative h
            else
                cos(phi) = sqrt(dist^2 - R^2) / dist
                h = <p_rel,  v_rel> + ||p_rel|| * ||v_rel|| * cos(phi)
        """

        theta = X[2, 0]
        v = X[3, 0]
        L_r = self.robot_spec['rear_ax_dist']

        obs_vel_x = 0
        obs_vel_y = 0

        # Combine radius R
        ego_dim = (obs[2][0] + self.robot_spec['body_width']) * beta   # Total collision radius

        # Compute relative position and velocity
        p_rel = np.array([[obs[0][0] - X[0, 0]], 
                        [obs[1][0] - X[1, 0]]])
        v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                        [obs_vel_y - v * np.sin(theta)]])  # Since the obstacle is static

        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)

        # Ensure safe buffer distance
        # if p_rel_mag <= ego_dim:
        #     h = -1e6
        #     dh_dx = np.zeros((1, 4))
        #     return h, dh_dx

        # Compute cos_phi safely
        cos_phi = np.sqrt(p_rel_mag**2 - ego_dim**2) / p_rel_mag

        # Compute h (C3BF)
        h = np.dot(p_rel.T, v_rel)[0, 0] + p_rel_mag * v_rel_mag * cos_phi
        
        # Compute ∂h/∂x (dh_dx)
        dh_dx = np.zeros((1, 4))

        # ∂h/∂x for x and y (position)
        dh_dx[0, 0] = -v_rel[0, 0] + (p_rel[0, 0] * v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2))
        dh_dx[0, 1] = -v_rel[1, 0] + (p_rel[1, 0] * v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2))

        # ∂h/∂theta
        dh_dx[0, 2] = v * (p_rel[0, 0] * np.sin(theta) - p_rel[1, 0] * np.cos(theta)) \
                    - np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * \
                    (v_rel[0, 0] * np.sin(theta) + v_rel[1, 0] * np.cos(theta))

        # ∂h/∂v
        dh_dx[0, 3] = -np.cos(theta) * p_rel[0, 0] - np.sin(theta) * p_rel[1, 0] \
                    + np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * \
                    (v_rel[0, 0] * np.cos(theta) + v_rel[1, 0] * np.sin(theta))

        print(f"dhdx: {dh_dx}")
        return h, dh_dx


    # def collision_cone_barrier(self, X, obs, beta=2.0):
    #     """
    #     Compute a Collision Cone Control Barrier Function for the Kinematic Bicycle (continous time).
        
    #     X: [x, y, theta, v]
    #     return: (h, h_dot, dh_dot_dx)
    #         h           : scalar, the C3BF value
    #         h_dot       : scalar, time derivative of h under f(x) and g(x)u

    #     The barrier is of "relative degree of 1" if done in velocity space, but for a kinematic bicycle, we can follow a pattern similar to agent_barrier:
    #         h_dot = ∂h/∂x ⋅ f(x) + ∂h_dot/∂x ⋅ g(x) ⋅ u

    #     Define h from the collision cone idea:
    #         p_rel = [obs_x - x, obs_y - y]
    #         v_rel = [-v_cos(theta), -v_sin(theta)] (since obstacle is static)
    #         dist = ||p_rel||
    #         R = robot_radius + obs_r (combined)
    #         inside = dist^2 - R^2

    #         if inside <= 0 -> already in collision possibility -> big negative h
    #         else
    #             cos(phi) = sqrt(dist^2 - R^2) / dist
    #             h = <p_rel,  v_rel> + ||p_rel|| * ||v_rel|| * cos(phi)
    #     """

    #     theta = X[2, 0]
    #     v = X[3, 0]
    #     L_r = self.robot_spec['rear_ax_dist']

    #     obs_vel_x = 0
    #     obs_vel_y = 0

    #     # Combine radius R
    #     # ego_dim = (obs[2][0] +  0.3]) # (* beta) # max(c1,c2) + robot_width/2
    #     ego_dim = obs[2][0] + self.robot_spec['body_width'] # (* beta) # max(c1,c2) + robot_width/2

    #     p_rel = np.array([[obs[0][0] - X[0, 0]], 
    #                       [obs[1][0] - X[1, 0]]])
    #     v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
    #                       [obs_vel_y - v * np.sin(theta)]]) # since obstacle is static
    #     # v_rel = (c_x_dot - v * np.cos(theta), c_y_dot - v * np.sin(theta))
    #     print(f"p_rel: {p_rel}, v_rel: {v_rel}")
    #     p_rel_mag = np.linalg.norm(p_rel)
    #     v_rel_mag = np.linalg.norm(v_rel)
        
    #     # check for safe buffer distance
    #     # if p_rel_mag <= ego_dim:
    #     #      h = -1e6
    #     #      Lf_h = -1e6
    #     #      Lg_h = np.zeros((1, 2))
    #     #      return h, Lf_h, Lg_h
        
    #     # Calculate cos_phi safely
    #     cos_phi = np.sqrt(p_rel_mag**2 - ego_dim**2) / p_rel_mag
    #     # cos_phi = np.clip(cos_phi, 0, 1) # 0 < cos(phi) < 1

    #     # print(f"cos_phi: {cos_phi}")

    #     # Compute h
    #     h = np.dot(p_rel.T, v_rel)[0, 0] + p_rel_mag * v_rel_mag * cos_phi
    #     # h = p_rel.T @ v_rel + p_rel_mag * v_rel_mag * cos_phi
    
    #     p_rel_x = p_rel[0, 0]
    #     p_rel_y = p_rel[1, 0]
    #     v_rel_x = v_rel[0, 0]
    #     v_rel_y = v_rel[1, 0]

    #     # print(f"p_rel_mag: {p_rel_mag}, ego_dim: {ego_dim}, diff: {p_rel_mag**2 - ego_dim**2}")
    #     print(f"sqrt: {np.sqrt(p_rel_mag**2 - ego_dim**2)}")
    #     h_dot_const = (v_rel_mag**2 + # from h_dot1
    #                    v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2 ) * p_rel_x * v_rel_x + v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2) * p_rel_y * v_rel_y) # from h_dot4
    #     h_dot_acc = (-p_rel_x * np.cos(theta) + -p_rel_y * np.sin(theta) + # from h_dot2
    #                   -np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * v_rel_x * np.cos(theta) - np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * v_rel_y * np.sin(theta)) # from h_dot3
    #     h_dot_beta = (v * np.sin(theta) * v_rel_x - v * np.cos(theta) * v_rel_y + # from h_dot1
    #                    p_rel_x * v**2 / L_r * np.sin(theta) - p_rel_y * v**2 / L_r * np.cos(theta) + # from h_dot2
    #                      np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * v_rel_x * v**2 / L_r * np.sin(theta) - np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * v_rel_y * v**2 / L_r * np.cos(theta) + # from h_dot3
    #                      v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2) * p_rel_x * v * np.sin(theta) - v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2) * p_rel_y * v * np.cos(theta)) # from h_dot4 

    #     # Compute Lfh, Lgh
    #     Lf_h = h_dot_const
    #     Lg_h = np.array([[h_dot_acc, h_dot_beta]])
    #     print(f"h: {h}, Lf_h: {Lf_h}, Lg_h: {Lg_h}")
        
    #     return h, Lf_h, Lg_h