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
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        return X

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
    
    def backup_input(self, X, k_a=1.0):
        """
        Using stop() function as backup policy
        """
        return self.stop(X, k_a=k_a)
    
    #### Backup policy for occlusion-aware adversaries #########
    def backup_input_occlusion(self, X, occlusion_scenarios, k_d=1.0, k_occ=1.0):
        """
        Occlusion-aware backup policy:
        - By default, apply velocity damping.
        - If there is a velocity component towards an occluded-unsafe direction, push the velocity away from that direction.
        """
        v = X[2:4, 0].astype(float)
        u = -k_d * v  # base damping

        if occlusion_scenarios:
            # Collect all facet normals from every occlusion scenario
            A_stack = []
            for sc in occlusion_scenarios:
                A = sc.get('A', None)
                if A is not None and A.size > 0:
                    A_stack.append(A)
            if len(A_stack) > 0:
                A_all = np.vstack(A_stack)  # (M_tot, 2)
                # soft aggrgation: use the average normal (signs are already chosen so that they point toward the unsafe side)
                n = -A_all.mean(axis=0)
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-6:
                    n = n / n_norm  # normalized unsafe-direction normal
                    v_dot_n = float(v @ n)
                    # If v_dot_n <0 means along the unsafe direction
                    if v_dot_n > 0.0:
                        u -= k_occ * v_dot_n * n

        # saturation
        a_max = float(self.robot_spec.get('a_max', 1.0))
        norm_u = np.linalg.norm(u)
        if norm_u > a_max > 0.0:
            u = u * (a_max / norm_u)

        return u.reshape(2, 1)

    def f_cl(self, X, occlusion_scenarios=None):
        """
        System dynamics as using backup policy u_b (Closed-Loop)
        """
        if occlusion_scenarios:
            u_b = self.backup_input_occlusion(X, occlusion_scenarios)
        else:
            u_b = self.backup_input(X)
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

    def h_b_stop(self, X):
        """
        Define Backup Set h_b(x) >= 0. (S_0)
        h_b = v_max^2 - (vx^2 + vy^2) >= 0
        """
        v_sq = X[2, 0]**2 + X[3, 0]**2
        v_safe_sq = (0.1)**2
        return v_safe_sq - v_sq

    def grad_h_b_stop(self, X):
        """
        Gradient of h_b_stop
        """
        return np.array([[0, 0, -2 * X[2, 0], -2 * X[3, 0]]])