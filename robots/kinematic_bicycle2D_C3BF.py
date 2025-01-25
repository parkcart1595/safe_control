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
            c3bf: h(x) = <p_rel, v_rel> + ||p_rel|| * ||v_rel|| * cos(pi)
            
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
    
    def compute_beta(self, X, G, k_theta=0.5):
        """
        Compute the slip angle beta based on the current state X and goal G.
        
        Parameters:
            X: Current state [x, y, theta, v].
            G: Goal state [x, y, theta, v].
            k_theta: Gain for the heading angle correction.

        Returns:
            beta: Slip angle in radians.
        """
        delta_max = self.robot_spec['delta_max']
        
        # Heading angle error
        theta_d = np.arctan2(G[1, 0] - X[1, 0], G[0, 0] - X[0, 0])
        error_theta = angle_normalize(theta_d - X[2, 0])
        
        # Steering angle
        delta = np.clip(k_theta * error_theta, -delta_max, delta_max)
        
        # Slip angle
        beta = self.beta(delta)
        return beta

            
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
   
    def nominal_input(self, X, G, d_min=0.05, k_theta=0.5, k_a = 1.5, k_v=0.5):
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
        beta = self.beta(delta)  # Slip angle conversion
                
        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = min(k_v * distance * np.cos(error_theta), v_max)
            
        a = k_a * (v - X[3, 0])
        return np.array([a, beta]).reshape(-1, 1)
    
    def stop(self, X):
        return np.array([0, 0]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        return abs(X[3, 0]) < tol  

    def rotate_to(self, X, theta_des, k_theta=2.0):
        error_theta = angle_normalize(theta_des - X[2, 0])
        beta = k_theta * error_theta
        return np.array([0.0, beta]).reshape(-1, 1)

    def agent_barrier(self, X, obs, robot_radius, beta=1.1):
        '''Continuous Time High Order CBF'''
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius  # obs radius + robot radius

        h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta*d_min**2
        # Lgh is zero => relative degree is 2
        h_dot = 2 * (X[0:2] - obsX[0:2]).T @ (self.f(X)[0:2])

        df_dx = self.df_dx(X)
        dh_dot_dx = np.append((2 * self.f(X)[0:2]).T, np.array(
            [[0, 0]]), axis=1) + 2 * (X[0:2] - obsX[0:2]).T @ df_dx[0:2, :]
        return h, h_dot, dh_dot_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.1):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

        def h(x, obs, robot_radius, beta=1.25):
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
    
    def collision_cone_barrier(self, X, obs, robot_radius):
        """
        Compute a Collision Cone Control Barrier Function for the Kinematic Bicycle (continous time).
        
        obs: (obs_x, obs_y, obs_r)
            We'll treat the obstacle as a circle of radius obs_r (static).
            We also inflate by 'robot_radius' to get total R.
        X: [x, y, theta, v]
        return: (h, h_dot, dh_dot_dx)
            h           : scalar, the C3BF value
            h_dot       : scalar, time derivative of h under f(x) (i.e. Lf h)
            dh_dot_dx   : (1,4) array, derivative of h_dot wrt x => used to get Lg h = dh_dot_dx*g(x)

        The barrier is of "relative degree of 1" if done in velocity space, but for a kinematic bicycle, we can follow a pattern similar to agent_barrier:
            h_dot = ∂h/∂x ⋅ f(x)
            Lg h = ∂h_dot/∂x ⋅ g(x)

        Define h from the collision cone idea:
            p_rel = [obs_x - x, obs_y - y]
            v_rel = [-v_cos(theta), -v_sin(theta)] (since obstacle is static)
            dist = ||p_rel||
            R = robot_radius + obs_r (combined)
            inside = dist^2 - R^2

            if inside <= 0 -> already in collision -> big negative h
            else
                cos(phi) = sqrt(dist^2 - R^2) / dist
                h = <p_rel,  v_rel> + dist * ||v_rel|| * cos(phi)

        Then h_dot, dh_dot_dx are found by chain rule. Below we do a small numeric approach for partials.

        NOTE: 'beta_margin' can let you enlarge or shrink R -> R_eff = R * beta_margin if you want extra margin.

        """
        theta = X[2, 0]
        v = X[3, 0]
        L_r = self.robot_spec['rear_ax_dist']

        obsX = obs[0:2]
        
        # Combine radius
        ego_dim = obs[2][0] + robot_radius # max(c1,c2) + robot_width/2

        obs_x, obs_y, obs_r = obs

        # # Combine radius
        # R = (robot_radius + obs_r) * beta_margin

        p_rel = np.array([[obs[0][0] - X[0, 0]], 
                          [obs[1][0] - X[1, 0]]])
        v_rel = np.array([[-v * np.cos(theta)], 
                          [-v * np.sin(theta)]]) # since obstacle is static
        # v_rel = (c_x_dot - v * np.cos(theta), c_y_dot - v * np.sin(theta))

        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)
        cos_phi = np.sqrt(p_rel_mag**2 - ego_dim**2) / p_rel_mag

        # p_rel_dot = v_rel + beta * np.array([[v * np.sin(theta)], 
        #                                    [-v * np.cos(theta)]])
        # v_rel_dot = np.array([[-np.cos(theta), v * np.sin(theta)],
        #                      [-np.sin(theta), -v * np.cos(theta)]]) @ np.array([[a],
        #                                                                        [v / L_r * beta]])

        # Compute h
        h = np.dot(p_rel.T, v_rel)[0, 0] + p_rel_mag * v_rel_mag * cos_phi

        p_rel_x = p_rel[0, 0]
        p_rel_y = p_rel[1, 0]
        v_rel_x = v_rel[0, 0]
        v_rel_y = v_rel[1, 0]
        # p_rel_dot_x = p_rel_dot[0, 0]
        # p_rel_dot_y = p_rel_dot[1, 0]
        # v_rel_dot_x = v_rel_dot[0, 0]
        # v_rel_dot_y = v_rel_dot[1, 0]

        h_dot_const = (v_rel_mag**2 + # from h_dot1
                       v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2) * p_rel_x * v_rel_x + v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2) * p_rel_y * v_rel_y) # from h_dot4
        h_dot_acc = (-p_rel_x * np.cos(theta) + -p_rel_y * np.sin(theta) + # from h_dot2
                      -np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * v_rel_x * np.cos(theta) + -np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * v_rel_y * np.sin(theta)) # from h_dot3
        h_dot_beta = (v * np.sin(theta) * v_rel_x + -v * np.cos(theta) * v_rel_y + # from h_dot1
                       p_rel_x * v**2 / L_r * np.sin(theta) + p_rel_y * -v**2 / L_r * np.cos(theta) + # from h_dot2
                         np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * v_rel_x * v**2 / L_r * np.sin(theta) + -np.sqrt(p_rel_mag**2 - ego_dim**2) / v_rel_mag * v_rel_y * v**2 / L_r * np.cos(theta) + # from h_dot3
                         v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2) * v * np.sin(theta) + -v_rel_mag / np.sqrt(p_rel_mag**2 - ego_dim**2) * v * np.cos(theta)) # from h_dot4 

        # Compute Lfh, Lgh
        Lf_h = h_dot_const
        Lg_h = np.array([[h_dot_acc, h_dot_beta]])

        return h, Lf_h, Lg_h