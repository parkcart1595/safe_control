from robots.kinematic_bicycle2D import KinematicBicycle2D
import numpy as np
import casadi as ca

"""
It is based on the kinematic bicycle 2D model and overrides
only the continous and discrete-time CBF funcitions for collision cone CBF (C3BF) counterparts:
ref: asdfasd/C3BF/arxiv.com
"""
def wrap_to_pi(angle):
    """Wraps an angle in radians to the interval [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class KinematicBicycle2D_CNBFVO(KinematicBicycle2D):
    def __init__(self, dt, robot_spec):
        super().__init__(dt, robot_spec)
        # Parameters from the paper
        self.kappa_min = 0.05
        self.delta_min = 0.05
        # Corresponds to the linear class-K function alpha(h) = gamma * h
        self.gamma = 0.5

    def speed_barrier(self, X, obs, robot_radius, beta=1.05):
        """
        Computes the speed barrier h_v,i and its derivative components for a single obstacle.
        This version assumes ZERO OBSTACLE ACCELERATION and uses the specified 'obs' format.
        """
        # 1. Get vehicle and obstacle states
        p = X[0:2, 0]
        psi, v = X[2, 0], X[3, 0]

        # Use the specified obs format: [px, py, radius, vx, vy]
        p_i = obs[0:2]
        radius_i = obs[2]
        v_i_vec = np.zeros(2)
        if len(obs) > 3:
            v_i_vec = obs[3:5]
        
        v_i = np.linalg.norm(v_i_vec)
        psi_i = np.arctan2(v_i_vec[1], v_i_vec[0])
        
        # ASSUMPTION: Obstacle acceleration is zero, as requested.
        v_dot_i = 0.0
        psi_dot_i = 0.0

        # 2. Calculate intermediate terms
        r_i = p_i - p
        d_i = np.linalg.norm(r_i)
        
        # Total minimum distance (vehicle radius + obstacle radius)
        d_min_i = (robot_radius + radius_i) * beta
        
        if d_i <= d_min_i:
            d_i = d_min_i + 1e-6 # Add epsilon to prevent division by zero

        # Eq. (14): beta_i
        beta_i = np.arcsin(d_min_i / d_i)
        
        angle_r_i = np.arctan2(r_i[1], r_i[0])

        # Eq. (22) part 1: psi_cc_i
        psi_cc_i_plus = angle_r_i + beta_i
        psi_cc_i_minus = angle_r_i - beta_i

        # Eq. (23) part 1: phi_i
        phi_i_plus = np.pi - psi_i + psi_cc_i_plus
        phi_i_minus = np.pi - psi_i + psi_cc_i_minus
        
        # 3. Compute the four smooth components of h_v,i from Eq. (27)
        h_options = {
            ('+', '+'): v + v_i * np.sin(phi_i_plus) - self.kappa_min,
            ('+', '-'): v + v_i * np.sin(phi_i_minus) - self.kappa_min,
            ('-', '+'): v - v_i * np.sin(phi_i_plus) - self.kappa_min,
            ('-', '-'): v - v_i * np.sin(phi_i_minus) - self.kappa_min,
        }
        
        # 4. Find the "most active" constraint (minimum h value)
        active_key = min(h_options, key=h_options.get)
        h_v = h_options[active_key]
        
        k_char, j_char = active_key
        k = 1.0 if k_char == '+' else -1.0
        phi_j = phi_i_plus if j_char == '+' else phi_i_minus
        psi_cc_j = psi_cc_i_plus if j_char == '+' else psi_cc_i_minus

        # 5. Compute derivative of h_v based on Eq. (30)
        # The control input 'a' is u2. Its coefficient in dot(h_v) is always 1.
        L_g_hv = 1.0 
        
        # We need dot(psi_cc) for L_f_hv
        v_r = (v * np.array([np.cos(psi), np.sin(psi)])) - v_i_vec
        d_dot_i = -(r_i @ v_r) / d_i # From Eq. (19)
        beta_dot_i = (-d_min_i * d_dot_i) / (d_i * np.sqrt(d_i**2 - d_min_i**2))
        angle_r_dot_i = (r_i[0] * -v_r[1] - r_i[1] * -v_r[0]) / d_i**2
        psi_cc_dot_j = angle_r_dot_i + beta_dot_i if j_char == '+' else angle_r_dot_i - beta_dot_i
        
        # With zero acceleration, dot(v_i) and dot(psi_i) are zero, simplifying Eq. (30)
        L_f_hv = k * v_i * np.cos(phi_j) * psi_cc_dot_j

        A_v = L_g_hv
        b_v = L_f_hv + self.gamma * h_v

        return A_v, b_v

    def steering_barrier(self, X, obs, robot_radius, u2_safe):
        """
        Computes the steering barrier h_psi,i and its derivative components.
        This version is corrected, assumes ZERO OBSTACLE ACCELERATION, 
        and uses the specified 'obs' format.
        """
        # 1. Get vehicle and obstacle states
        p = X[0:2, 0]
        psi, v = X[2, 0], X[3, 0]

        p_i = obs[0:2]
        radius_i = obs[2]
        v_i_vec = np.zeros(2)
        if len(obs) > 3:
            v_i_vec = obs[3:5]

        v_i = np.linalg.norm(v_i_vec)
        psi_i = np.arctan2(v_i_vec[1], v_i_vec[0])
        
        # ASSUMPTION: Obstacle acceleration is zero.
        v_dot_i = 0.0
        psi_dot_i = 0.0

        r_i = p_i - p
        d_i = np.linalg.norm(r_i)
        d_min_i = robot_radius + radius_i
        
        if d_i <= d_min_i: d_i = d_min_i + 1e-6
            
        beta_i = np.arcsin(d_min_i / d_i)
        angle_r_i = np.arctan2(r_i[1], r_i[0])
        psi_cc_i_plus = angle_r_i + beta_i
        psi_cc_i_minus = angle_r_i - beta_i
        
        phi_i_plus = np.pi - psi_i + psi_cc_i_plus
        phi_i_minus = np.pi - psi_i + psi_cc_i_minus
        
        # 2. Calculate VO angles psi_vo,i from Eq. (22) and (23)
        # Clip the argument to arcsin to handle potential numerical precision issues
        arg_sin_plus = np.clip((v_i / v) * np.sin(phi_i_plus), -1.0, 1.0)
        arg_sin_minus = np.clip((v_i / v) * np.sin(phi_i_minus), -1.0, 1.0)
        
        vartheta_i_plus = np.arcsin(arg_sin_plus)
        vartheta_i_minus = np.arcsin(arg_sin_minus)

        psi_vo_i_plus = psi_cc_i_plus + vartheta_i_plus   # Left vertex of VO cone
        psi_vo_i_minus = psi_cc_i_minus + vartheta_i_minus # Right vertex of VO cone

        # 3. Calculate angular distances and find the active barrier (CORRECTED LOGIC)
        # These are the signed angular distances from our heading (psi) to the VO boundaries.
        delta_i_plus = wrap_to_pi(psi - psi_vo_i_plus)
        delta_i_minus = wrap_to_pi(psi - psi_vo_i_minus)

        # The active barrier is the one corresponding to the closest VO boundary, per Eq. (32).
        if abs(delta_i_plus) < abs(delta_i_minus):
            # Closest to the LEFT boundary ('+'). We need to steer RIGHT.
            # h_psi should be positive when safe (i.e., when psi > psi_vo_i_plus).
            h_psi = delta_i_plus - self.delta_min
            j_char = '+'
            phi_j, psi_cc_j = phi_i_plus, psi_cc_i_plus
            # Constraint: u1 - d(psi_vo+)/dt >= -gamma*h. So A=1, b=-d(psi_vo+)/dt + gamma*h
            sign = 1.0
        else:
            # Closest to the RIGHT boundary ('-'). We need to steer LEFT.
            # h_psi should be positive when safe (i.e., when psi < psi_vo_i_minus).
            h_psi = wrap_to_pi(psi_vo_i_minus - psi) - self.delta_min
            j_char = '-'
            phi_j, psi_cc_j = phi_i_minus, psi_cc_i_minus
            # Constraint: d(psi_vo-)/dt - u1 >= -gamma*h. So A=-1, b=d(psi_vo-)/dt + gamma*h
            sign = -1.0

        # 4. Compute derivative components based on Eq. (35)
        v_r = (v * np.array([np.cos(psi), np.sin(psi)])) - v_i_vec
        d_dot_i = -(r_i @ v_r) / d_i
        
        # Avoid sqrt of negative number from precision errors
        sqrt_di_arg = d_i**2 - d_min_i**2
        if sqrt_di_arg < 0: sqrt_di_arg = 0
        beta_dot_i = (-d_min_i * d_dot_i) / (d_i * np.sqrt(sqrt_di_arg) + 1e-6)
        
        angle_r_dot_i = (r_i[0] * -v_r[1] - r_i[1] * -v_r[0]) / (d_i**2 + 1e-6)
        psi_cc_dot_j = angle_r_dot_i + beta_dot_i if j_char == '+' else angle_r_dot_i - beta_dot_i

        sqrt_v_arg = v**2 - (v_i * np.sin(phi_j))**2
        if sqrt_v_arg < 0: sqrt_v_arg = 0
        sqrt_term = np.sqrt(sqrt_v_arg) + 1e-6
            
        # These terms collectively represent the time derivative of psi_vo_i
        # Eq (35) simplifies with dot(psi_i) = 0 and dot(v_i) = 0
        term1 = psi_cc_dot_j
        term2 = (psi_cc_dot_j) * (v_i * np.cos(phi_j)) / sqrt_term
        # Use the safe acceleration u2_safe from the first QP
        term3 = (-v_i * u2_safe) * np.sin(phi_j) / (v * sqrt_term)
        
        psi_vo_dot = term1 + term2 + term3

        # L_f_hpsi is the part of the derivative NOT dependent on our control input u1 (which is r)
        # For the left boundary: d(h)/dt = u1 - psi_vo_dot
        # For the right boundary: d(h)/dt = -u1 + psi_vo_dot
        L_f_hpsi = -psi_vo_dot if sign == 1.0 else psi_vo_dot
        
        # L_g_hpsi is the coefficient of our control input u1
        L_g_hpsi = 1.0 if sign == 1.0 else -1.0
        
        A_psi = L_g_hpsi
        b_psi = L_f_hpsi + self.gamma * h_psi

        return A_psi, b_psi