from robots.kinematic_bicycle2D import KinematicBicycle2D
import numpy as np
import casadi as ca

"""
It is based on the kinematic bicycle 2D model and overrides
only the continous and discrete-time CBF funcitions for collision cone CBF (C3BF) counterparts:
ref: asdfasd/C3BF/arxiv.com
"""

class KinematicBicycle2D_C3BF(KinematicBicycle2D):
    def __init__(self, dt, robot_spec):
        super().__init__(dt, robot_spec)

    def agent_barrier(self, X, obs, robot_radius, beta=1.0):
        """
        '''Continuous Time C3BF'''
        Compute a Collision Cone Control Barrier Function for the Kinematic Bicycle2D.

        The barrier's relative degree is "1"
            h_dot = ∂h/∂x ⋅ f(x) + ∂h/∂x ⋅ g(x) ⋅ u

        Define h from the collision cone idea:
            p_rel = [obs_x - x, obs_y - y]
            v_rel = [obs_x_dot-v_cos(theta), obs_y_dot-v_sin(theta)]
            dist = ||p_rel||
            R = robot_radius + obs_r
        """

        theta = X[2, 0]
        v = X[3, 0]
        
        ############### For nearest_obs setting
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
        v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                        [obs_vel_y - v * np.sin(theta)]])
    

        p_rel_x = p_rel[0, 0]
        p_rel_y = p_rel[1, 0]
        v_rel_x = v_rel[0, 0]
        v_rel_y = v_rel[1, 0]

        rot_angle = np.arctan2(p_rel_y, p_rel_x)

        # Rotation matrix for vrel vector **
        R = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                    [-np.sin(rot_angle),  np.cos(rot_angle)]])
        
        # Transform v_rel into the new coordinate frame
        v_rel_new = R @ v_rel

        v_rel_new_x = v_rel_new[0, 0]
        v_rel_new_y = v_rel_new[1, 0]

        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)

        # Compute d_safe safely
        # eps = 1e-6
        eps = 1e-6
        # d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)
        d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)
        # Penalty term
        k_lamda, k_mu = 0.3, 1.0
        d_pen = v_rel_mag
        # vel_pen = a * v_rel_mag
        lamda = k_lamda * np.sqrt(d_safe) / v_rel_mag # same as 1/tan(phi)
        mu = k_mu * (np.sqrt(d_safe) - d_pen)
        
        # Barrier function h(x)
        h = v_rel_new_x + lamda * (v_rel_new_y**2) + mu
        # print(f"v_rel_new_x: {v_rel_new_x} | lamda: {lamda} | mu: {mu} | rot_angle: {rot_angle}")
        # print(h)
        
        # Compute h (C3BF)
        dh_dx = np.zeros((1, 4))

        dh_dx[0, 0] = - k_lamda * p_rel_x / (v_rel_mag * np.sqrt(d_safe)) * (v_rel_new_y)**2 - k_lamda * p_rel_x / np.sqrt(d_safe)
        dh_dx[0, 1] = - k_lamda * p_rel_y / (v_rel_mag * np.sqrt(d_safe)) * (v_rel_new_y)**2 - k_lamda * p_rel_y / np.sqrt(d_safe)
        dh_dx[0, 2] = np.cos(rot_angle) * v * np.sin(theta) - np.sin(rot_angle) * v * np.cos(theta) - k_lamda * np.sqrt(d_safe) / v_rel_mag**3 * (v_rel_x * v * np.sin(theta) - v_rel_y * v * np.cos(theta)) * (v_rel_new_y)**2 + 2 * k_lamda * np.sqrt(d_safe) / v_rel_mag * v_rel_new_y * (-np.sin(rot_angle) * v * np.sin(theta) - np.cos(rot_angle) * v * np.cos(theta))
        dh_dx[0, 3] = -np.cos(rot_angle) * np.cos(theta) - np.sin(rot_angle) * np.sin(theta) - k_lamda * np.sqrt(d_safe) / v_rel_mag**3 * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta)) * v_rel_new_y**2 + 2 * k_lamda * np.sqrt(d_safe) / v_rel_mag * v_rel_new_y * (np.sin(rot_angle) * np.cos(theta) - np.cos(rot_angle) * np.sin(theta)) - k_mu * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta)) / v_rel_mag

        # dh_dx[0, 0] = - k_lamda * p_rel_x / v_rel_mag / p_rel_mag * (v_rel_new_y)**2 - k_lamda * p_rel_x
        # dh_dx[0, 1] = - k_lamda * p_rel_y / v_rel_mag / p_rel_mag * (v_rel_new_y)**2 - k_lamda * p_rel_y
        # dh_dx[0, 2] = np.cos(rot_angle) * v * np.sin(theta) - np.sin(rot_angle) * v * np.cos(theta) + 2 * k_lamda * d_safe / v_rel_mag * v_rel_new_y * (-np.sin(rot_angle) * v * np.sin(theta) - np.cos(rot_angle) * v * np.cos(theta))
        # dh_dx[0, 3] = -np.cos(rot_angle) * np.cos(theta) - np.sin(rot_angle) * np.sin(theta) - k_lamda * d_safe / v_rel_mag**3 * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta)) * v_rel_new_y**2 + 2 * k_lamda * d_safe / v_rel_mag * v_rel_new_y * (np.sin(rot_angle) * np.cos(theta) - np.cos(rot_angle) * np.sin(theta))
        
        return h, dh_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.0):
        '''Discrete Time Q3BF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k, casadi=True)

        def h(x, obs, robot_radius, beta=1.0):
            theta = x[2, 0]
            v = x[3, 0]

            # Check if obstacles have velocity components (static or moving)
            if obs.shape[0] > 3:
                obs_vel_x = obs[3][0]
                obs_vel_y = obs[4][0]
            else:
                obs_vel_x = 0.0
                obs_vel_y = 0.0
            
            # Combine radius R
            ego_dim = (obs[2][0] + robot_radius) * beta   # Total collision radius

            # Compute relative position and velocity
            p_rel = ca.vertcat(obs[0][0] - x[0, 0], obs[1][0] - x[1, 0])
            v_rel = ca.vertcat(obs_vel_x - v * ca.cos(theta), obs_vel_y - v * ca.sin(theta))

            # Compute the rotation angle: align p_rel with y-axis
            rot_angle = ca.atan2(p_rel[1], p_rel[0])

            # Rotation matrix for transforming to the new coordinate frame:
            # Using R(-rot_angle) to rotate vectors such that p_rel aligns with the y-axis
            R = ca.vertcat( 
                ca.horzcat(ca.cos(rot_angle), ca.sin(rot_angle)),
                ca.horzcat(-ca.sin(rot_angle), ca.cos(rot_angle))
            )

            # Transform v_rel into the new coordinate frame
            v_rel_new = ca.mtimes(R, v_rel)

            p_rel_mag = ca.norm_2(p_rel)
            v_rel_mag = ca.norm_2(v_rel)
            
            eps = 1e-6
            d_safe = ca.fmax(p_rel_mag**2 - ego_dim**2, eps)

            k_lamda, k_mu = 0.36, 0.45
            
            lamda = k_lamda * ca.sqrt(d_safe) / v_rel_mag
            mu = k_mu * (ca.sqrt(d_safe) - v_rel_mag)

            # Compute h
            h = v_rel_new[0] + lamda * v_rel_new[1]**2 + mu

            return h

        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)
        
        d_h = h_k1 - h_k

        return h_k, d_h