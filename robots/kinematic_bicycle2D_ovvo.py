# In your robot class file (e.g., robot.py)
from robots.kinematic_bicycle2D import KinematicBicycle2D
import numpy as np

class KinematicBicycle2D_OVVO(KinematicBicycle2D):
    def __init__(self, dt, robot_spec):
        super().__init__(dt, robot_spec)

    def agent_barrier(self, X, obs, robot_radius, beta=1.1):
        """
        Computes and returns the results for both OVVO-inspired barrier functions.
        """
        # Get the state X from the class instance
        # X = self.X

        # Call the two barrier functions
        clearance_results = self.clearance_barrier(X, obs, robot_radius, d_min=0.1)
        passtime_results = self.passtime_barrier(X, obs, robot_radius, t_min=5.0)
        
        return clearance_results, passtime_results

    # --- Paste the two new barrier functions here ---

    def clearance_barrier(self, X, obs, robot_radius, d_min=0.1, beta=1.1):
        """
        Computes a Clearance-based Control Barrier Function inspired by OVVO.
        """
        # ... (full code from the previous response) ...
        theta = X[2, 0]
        v = X[3, 0]
        obs_vel_x = obs[3] if obs.shape[0] > 3 else 0.0
        obs_vel_y = obs[4] if obs.shape[0] > 4 else 0.0
        p_rel = np.array([[obs[0] - X[0, 0]], [obs[1] - X[1, 0]]])
        v_rel = np.array([[obs_vel_x - v * np.cos(theta)], [obs_vel_y - v * np.sin(theta)]])
        p_rel_x, p_rel_y = p_rel[0, 0], p_rel[1, 0]
        v_rel_x, v_rel_y = v_rel[0, 0], v_rel[1, 0]
        eps = 1e-6
        v_rel_mag = np.linalg.norm(v_rel)
        cross_prod_mag = np.abs(v_rel_x * p_rel_y - v_rel_y * p_rel_x)
        d_v = cross_prod_mag / (v_rel_mag + eps)
        h = d_v - (robot_radius + obs[2]) * beta - d_min
        dh_dx = np.zeros((1, 4))
        common_term_numerator = v_rel_x * p_rel_y - v_rel_y * p_rel_x
        S = np.sign(common_term_numerator)
        dN_dx = S * (-v_rel_y)
        dN_dy = S * (v_rel_x)
        dD_dx, dD_dy = 0, 0
        dh_dx[0, 0] = (dN_dx * v_rel_mag - cross_prod_mag * dD_dx) / (v_rel_mag**2 + eps)
        dh_dx[0, 1] = (dN_dy * v_rel_mag - cross_prod_mag * dD_dy) / (v_rel_mag**2 + eps)
        dv_rel_x_dtheta, dv_rel_y_dtheta = v * np.sin(theta), -v * np.cos(theta)
        dN_dtheta = S * (dv_rel_x_dtheta * p_rel_y - dv_rel_y_dtheta * p_rel_x)
        dD_dtheta = (v_rel_x * dv_rel_x_dtheta + v_rel_y * dv_rel_y_dtheta) / (v_rel_mag + eps)
        dh_dx[0, 2] = (dN_dtheta * v_rel_mag - cross_prod_mag * dD_dtheta) / (v_rel_mag**2 + eps)
        dv_rel_x_dv, dv_rel_y_dv = -np.cos(theta), -np.sin(theta)
        dN_dv = S * (dv_rel_x_dv * p_rel_y - dv_rel_y_dv * p_rel_x)
        dD_dv = (v_rel_x * dv_rel_x_dv + v_rel_y * dv_rel_y_dv) / (v_rel_mag + eps)
        dh_dx[0, 3] = (dN_dv * v_rel_mag - cross_prod_mag * dD_dv) / (v_rel_mag**2 + eps)
        return h, dh_dx

    def passtime_barrier(self, X, obs, robot_radius, t_min=1.0):
        """
        Computes a Pass-Time-based Control Barrier Function inspired by OVVO.
        """
        # ... (full code from the previous response) ...
        theta = X[2, 0]
        v = X[3, 0]
        obs_vel_x = obs[3] if obs.shape[0] > 3 else 0.0
        obs_vel_y = obs[4] if obs.shape[0] > 4 else 0.0
        p_rel = np.array([[obs[0] - X[0, 0]], [obs[1] - X[1, 0]]])
        v_rel = np.array([[obs_vel_x - v * np.cos(theta)], [obs_vel_y - v * np.sin(theta)]])
        p_rel_x, p_rel_y = p_rel[0, 0], p_rel[1, 0]
        v_rel_x, v_rel_y = v_rel[0, 0], v_rel[1, 0]
        eps = 1e-6
        v_rel_mag_sq = v_rel_x**2 + v_rel_y**2
        p_rel_dot_v_rel = p_rel_x * v_rel_x + p_rel_y * v_rel_y
        if p_rel_dot_v_rel >= 0:
            return 100.0, np.zeros((1, 4))
        t_ca = -p_rel_dot_v_rel / (v_rel_mag_sq + eps)
        h = t_ca - t_min
        dh_dx = np.zeros((1, 4))
        dN_dx, dN_dy = v_rel_x, v_rel_y
        dD_dx, dD_dy = 0, 0
        dh_dx[0, 0] = (dN_dx * v_rel_mag_sq - (-p_rel_dot_v_rel) * dD_dx) / (v_rel_mag_sq**2 + eps)
        dh_dx[0, 1] = (dN_dy * v_rel_mag_sq - (-p_rel_dot_v_rel) * dD_dy) / (v_rel_mag_sq**2 + eps)
        dv_rel_x_dtheta, dv_rel_y_dtheta = v * np.sin(theta), -v * np.cos(theta)
        dN_dtheta = -(p_rel_x * dv_rel_x_dtheta + p_rel_y * dv_rel_y_dtheta)
        dD_dtheta = 2 * (v_rel_x * dv_rel_x_dtheta + v_rel_y * dv_rel_y_dtheta)
        dh_dx[0, 2] = (dN_dtheta * v_rel_mag_sq - (-p_rel_dot_v_rel) * dD_dtheta) / (v_rel_mag_sq**2 + eps)
        dv_rel_x_dv, dv_rel_y_dv = -np.cos(theta), -np.sin(theta)
        dN_dv = -(p_rel_x * dv_rel_x_dv + p_rel_y * dv_rel_y_dv + (-1*np.cos(theta)*v_rel_x - np.sin(theta)*v_rel_y))
        dD_dv = 2 * (v_rel_x * dv_rel_x_dv + v_rel_y * dv_rel_y_dv)
        dh_dx[0, 3] = (dN_dv * v_rel_mag_sq - (-p_rel_dot_v_rel) * dD_dv) / (v_rel_mag_sq**2 + eps)
        return h, dh_dx