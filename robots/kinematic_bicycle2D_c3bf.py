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
        
        # # Check if obstacles have velocity components (static or moving)
        # if obs.shape[0] > 3:
        #     obs_vel_x = obs[3]
        #     obs_vel_y = obs[4]

        # else:
        #     obs_vel_x = 0.0
        #     obs_vel_y = 0.0

        # # Combine radius R
        # ego_dim = (obs[2] + robot_radius) * beta # Total collision safe radius

        # # Compute relative position and velocity
        # p_rel = np.array([[obs[0] - X[0, 0]], 
        #                 [obs[1] - X[1, 0]]])
        # v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
        #                 [obs_vel_y - v * np.sin(theta)]])
        
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
        v_rel_new_mag = np.linalg.norm(v_rel_new)

        # Compute d_safe safely
        eps = 1e-6
        d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)

        # Penalty term
        # a, b = 0.01, 1.5
        a, b = 1.0, 2/3
        # vel_pen = a * v_rel_mag
        slope_pen = a * np.sqrt(d_safe) / v_rel_mag # same as 1/tan(phi)
        dist_pen = b * np.sqrt(d_safe)
        
        # Barrier function h(x)
        h = v_rel_new_x + slope_pen * (v_rel_new_y**2) + dist_pen
        print(f"v_rel_new_x: {v_rel_new_x} | slope_pen : {slope_pen} | dist_pen: {dist_pen} | rot_angle: {rot_angle}")
        print(h)
        
        # Compute h (C3BF)
        dh_dx = np.zeros((1, 4))

        # dh_dx[0, 0] = - a * p_rel_x / (2 * ego_dim * v_rel_mag * np.sqrt(d_safe)) * (v_rel_new_y)**2 - b * p_rel_x * v_rel_mag / (2 * ego_dim * np.sqrt(d_safe))
        # dh_dx[0, 1] = - a * p_rel_y / (2 * ego_dim * v_rel_mag * np.sqrt(d_safe)) * (v_rel_new_y)**2 - b * p_rel_y * v_rel_mag / (2 * ego_dim * np.sqrt(d_safe))
        # dh_dx[0, 2] = np.cos(rot_angle) * v * np.sin(theta) - np.sin(rot_angle) * v * np.cos(theta) - a * np.sqrt(d_safe) / (2 * ego_dim * v_rel_mag**3) * (v_rel_x * v * np.sin(theta) - v_rel_y * v * np.cos(theta)) * (v_rel_new_y)**2 + a * np.sqrt(d_safe) / (ego_dim * v_rel_mag) * v_rel_new_y * (-np.sin(rot_angle) * v * np.sin(theta) - np.cos(rot_angle) * v * np.cos(theta)) + b * np.sqrt(d_safe) / (2 * ego_dim * v_rel_mag) * (v_rel_x * v * np.sin(theta) - v_rel_y * v * np.cos(theta))
        # dh_dx[0, 3] = -np.cos(rot_angle) * np.cos(theta) - np.sin(rot_angle) * np.sin(theta) - a * np.sqrt(d_safe) / (2 * ego_dim * v_rel_mag**3) * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta)) * v_rel_new_y**2 + a * np.sqrt(d_safe) / (ego_dim * v_rel_mag) * v_rel_new_y * (np.sin(rot_angle) * np.cos(theta) - np.cos(rot_angle) * np.sin(theta)) + b * np.sqrt(d_safe) / (2 * ego_dim * v_rel_mag) * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta))

        dh_dx[0, 0] = - a * p_rel_x / (v_rel_mag * np.sqrt(d_safe)) * (v_rel_new_y)**2 - b * p_rel_x / np.sqrt(d_safe)
        dh_dx[0, 1] = - a * p_rel_y / (v_rel_mag * np.sqrt(d_safe)) * (v_rel_new_y)**2 - b * p_rel_y / np.sqrt(d_safe)
        dh_dx[0, 2] = np.cos(rot_angle) * v * np.sin(theta) - np.sin(rot_angle) * v * np.cos(theta) - a * np.sqrt(d_safe) / v_rel_mag**3 * (v_rel_x * v * np.sin(theta) - v_rel_y * v * np.cos(theta)) * (v_rel_new_y)**2 + 2 * a * np.sqrt(d_safe) / v_rel_mag * v_rel_new_y * (-np.sin(rot_angle) * v * np.sin(theta) - np.cos(rot_angle) * v * np.cos(theta))
        dh_dx[0, 3] = -np.cos(rot_angle) * np.cos(theta) - np.sin(rot_angle) * np.sin(theta) - a * np.sqrt(d_safe) / v_rel_mag**3 * (v - obs_vel_x * np.cos(theta) - obs_vel_y * np.sin(theta)) * v_rel_new_y**2 + 2 * a * np.sqrt(d_safe) / v_rel_mag * v_rel_new_y * (np.sin(rot_angle) * np.cos(theta) - np.cos(rot_angle) * np.sin(theta))
        return h, dh_dx

    # def agent_barrier(self, X, obs, robot_radius, beta=1.0):
    #     """
    #     '''Continuous Time C3BF'''
    #     Compute a Collision Cone Control Barrier Function for the Kinematic Bicycle2D.

    #     The barrier's relative degree is "1"
    #         h_dot = ∂h/∂x ⋅ f(x) + ∂h/∂x ⋅ g(x) ⋅ u

    #     Define h from the collision cone idea:
    #         p_rel = [obs_x - x, obs_y - y]
    #         v_rel = [obs_x_dot-v_cos(theta), obs_y_dot-v_sin(theta)]
    #         dist = ||p_rel||
    #         R = robot_radius + obs_r
    #     """

    #     theta = X[2, 0]
    #     v = X[3, 0]
        
    #     # Check if obstacles have velocity components (static or moving)
    #     if obs.shape[0] > 3:
    #         obs_vel_x = obs[3]
    #         obs_vel_y = obs[4]

    #     else:
    #         obs_vel_x = 0.0
    #         obs_vel_y = 0.0

    #     # Combine radius R
    #     ego_dim = (obs[2] + robot_radius) * beta # Total collision safe radius

    #     # Compute relative position and velocity
    #     p_rel = np.array([[obs[0] - X[0, 0]], 
    #                     [obs[1] - X[1, 0]]])
    #     v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
    #                     [obs_vel_y - v * np.sin(theta)]])

    #     p_rel_x = p_rel[0, 0]
    #     p_rel_y = p_rel[1, 0]
    #     v_rel_x = v_rel[0, 0]
    #     v_rel_y = v_rel[1, 0]

    #     p_rel_mag = np.linalg.norm(p_rel)
    #     v_rel_mag = np.linalg.norm(v_rel)

    #     # Compute cos_phi safely for c3bf
    #     eps = 1e-6
    #     cal_max = np.maximum(p_rel_mag**2 - ego_dim**2, eps)
    #     sqrt_term = np.sqrt(cal_max)
    #     cos_phi = sqrt_term / (p_rel_mag + eps)
        
    #     # Compute h
    #     h = np.dot(p_rel.T, v_rel)[0, 0] + p_rel_mag * v_rel_mag * cos_phi

    #     # Compute dh_dx
    #     dh_dx = np.zeros((1, 4))
    #     dh_dx[0, 0] = -v_rel_x - v_rel_mag * p_rel_x / (sqrt_term + eps) 
    #     dh_dx[0, 1] = -v_rel_y - v_rel_mag * p_rel_y / (sqrt_term + eps)
    #     dh_dx[0, 2] =  v * np.sin(theta) * p_rel_x - v * np.cos(theta) * p_rel_y + (sqrt_term + eps) / v_rel_mag * (v * (obs_vel_x * np.sin(theta) - obs_vel_y * np.cos(theta)))
    #     dh_dx[0, 3] = -np.cos(theta) * p_rel_x -np.sin(theta) * p_rel_y + (sqrt_term + eps) / v_rel_mag * (v - (obs_vel_x * np.cos(theta) + obs_vel_y * np.sin(theta)))

    #     return h, dh_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time C3BF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k, casadi=True)

        def h(x, obs, robot_radius, beta=1.01):
            '''Computes C3BF h(x) = <p_rel, v_rel> + ||p_rel||*||v_rel||*cos(phi)'''
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

            p_rel_mag = ca.norm_2(p_rel)
            v_rel_mag = ca.norm_2(v_rel)

            # Compute h
            h = (p_rel.T @ v_rel)[0, 0] + p_rel_mag * v_rel_mag * ca.sqrt(ca.fmax(p_rel_mag**2 - ego_dim**2, 0)) / p_rel_mag
                
            return h

        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)
        
        d_h = h_k1 - h_k

        return h_k, d_h