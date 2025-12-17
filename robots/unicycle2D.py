import numpy as np
import casadi as ca
import cvxpy as cp

"""
Created on July 14h, 2024
@author: Taekyung Kim

@description: 
Kinematic unicycle model for CBF-QP and MPC-CBF (casadi)
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

class Unicycle2D:
    
    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, theta]
            U: [v, omega]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2 - sigma(s)
            relative degree: 1
        '''
        self.dt = dt
        self.robot_spec = robot_spec # not used in this model
      
        # for exp (CBF for unicycle)
        self.k1 = 0.5 #=#1.0
        self.k2 = 1.8 #0.5

        self.robot_spec.setdefault('v_max', 1.0)
        self.robot_spec.setdefault('w_max', 0.5)
        # controller config
        self.u_dim = 2
        self._occ_barrier_fn = None

    def f(self, X, casadi=False):
        X3 = X[:3]
        if casadi:
            return ca.vertcat([
                0, 
                0, 
                0
            ])
        else:
            return np.array([0,0,0]).reshape(-1,1)
    
    def g(self, X, casadi=False):
        X3 = X[:3]
        if casadi:
            g = ca.SX.zeros(3, 2)
            g[0, 0] = ca.cos(X3[2,0])
            g[1, 0] = ca.sin(X3[2,0])
            g[2, 1] = 1
            return g
        else:
            return np.array([ [ np.cos(X3[2,0]), 0],
                            [ np.sin(X3[2,0]), 0],
                            [0, 1] ]) 
         
    def step(self, X, U): 
        X_new = X.copy()
        base = X[:3]
        base = base + ( self.f(base) + self.g(base) @ U )*self.dt
        base[2,0] = angle_normalize(base[2,0])
        X_new[:3] = base
        return X_new

    def nominal_input(self, X, G, d_min = 0.05, k_omega = 2.0, k_v = 1.0):
        '''
        nominal input for CBF-QP
        '''
        G = np.copy(G.reshape(-1,1)) # goal state

        distance = max(np.linalg.norm( X[0:2,0]-G[0:2,0] ) - d_min, 0.05)
        theta_d = np.arctan2(G[1,0]-X[1,0],G[0,0]-X[0,0])
        error_theta = angle_normalize( theta_d - X[2,0] )

        omega = k_omega * error_theta   
        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = k_v*( distance )*np.cos( error_theta )

        return np.array([v, omega]).reshape(-1,1)
    
    def stop(self, X):
        return np.array([0,0]).reshape(-1,1)
    
    def has_stopped(self, X):
        # unicycle can always stop immediately
        return True

    def rotate_to(self, X, theta_des, k_omega = 2.0):
        error_theta = angle_normalize( theta_des - X[2,0] )
        omega = k_omega * error_theta
        return np.array([0.0, omega]).reshape(-1,1)
    
    def sigma(self,s):
        #print("s", s)
        return self.k2 * (np.exp(self.k1-s)-1)/(np.exp(self.k1-s)+1)
    
    def sigma_der(self,s):
        return - self.k2 * np.exp(self.k1-s)/( 1+np.exp( self.k1-s ) ) * ( 1 - self.sigma(s)/self.k2 )
    
    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius # obs radius + robot radius

        theta = X[2,0]

        h = np.linalg.norm( X[0:2] - obsX[0:2] )**2 - beta*d_min**2   
        s = ( X[0:2] - obsX[0:2]).T @ np.array( [np.cos(theta),np.sin(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        
        der_sigma = self.sigma_der(s)
        # [dh/dx, dh/dy, dh/dtheta]^T
        dh_dx = np.append( 
                    2*( X[0:2] - obsX[0:2] ).T - der_sigma * ( np.array([ [np.cos(theta), np.sin(theta)] ]) ),
                    - der_sigma * ( -np.sin(theta)*( X[0,0]-obsX[0,0] ) + np.cos(theta)*( X[1,0] - obsX[1,0] ) ),
                     axis=1)
        # print(h)
        # print(dh_dx)
        return h, dh_dx
        
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta = 1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)

        def h(x, obs, robot_radius, beta = 1.01):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            y_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 - beta*d_min**2
            return h

        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        return h_k, d_h
    
    # === Backup CBF support ===
    def input_constraints(self, u_var):
        v_max = float(self.robot_spec.get('v_max', np.inf))
        w_max = float(self.robot_spec.get('w_max', np.inf))
        return [cp.abs(u_var[0]) <= v_max,
                cp.abs(u_var[1]) <= w_max]

    def set_occ_barrier_fn(self, fn):
        self._occ_barrier_fn = fn

    def backup_input(self, X):
        return self.stop(X)

    def backup_input_occlusion(self, X, occlusion_scenarios=None, t=None, k_d=1.0, k_occ=1.0):
        # simple occlusion-aware backup:
        # - steer away from occlusion normals (risk_normal_vec)
        # - throttle proportional to heading alignment
        if occlusion_scenarios is None or len(occlusion_scenarios) == 0:
            return self.stop(X)

        dir_vecs = []
        for sc in occlusion_scenarios:
            rv = sc.get('risk_normal_vec', None)
            if rv is not None and rv.size >= 2:
                dir_vecs.append(rv.reshape(-1, 2).mean(axis=0))
        if not dir_vecs:
            return self.stop(X)

        dir_avg = np.mean(dir_vecs, axis=0)
        if np.linalg.norm(dir_avg) < 1e-9:
            return self.stop(X)

        dir_unit = dir_avg / np.linalg.norm(dir_avg)
        theta = float(X[2, 0])
        heading = np.array([np.cos(theta), np.sin(theta)])

        target_yaw = np.arctan2(dir_unit[1], dir_unit[0])
        yaw_err = angle_normalize(target_yaw - theta)

        v_max = float(self.robot_spec.get('v_max', 1.0))
        w_max = float(self.robot_spec.get('w_max', 0.5))

        # forward speed scales with alignment to safe direction
        align = max(0.0, heading @ dir_unit)
        v_cmd = min(v_max, k_occ * align * v_max)

        w_cmd = np.clip(k_d * yaw_err, -w_max, w_max)

        return np.array([v_cmd, w_cmd]).reshape(-1, 1)

    def f_cl(self, X, occlusion_scenarios=None, t=None):
        u_b = self.backup_input(X)
        xdot = self.f(X) + self.g(X) @ u_b  # shape (3,1)
        n = X.shape[0]
        if n > 3:
            pad = np.zeros((n - 3, 1))
            xdot = np.vstack([xdot, pad])
        return xdot

    def F_cl(self, X, occlusion_scenarios=None, t=None):
        n = X.shape[0]
        F = np.zeros((n, n))
        if hasattr(self, "df_dx"):
            base = self.df_dx(X)
            F[:base.shape[0], :base.shape[1]] = base
        return F

    def simulate_backup_trajectory(self, x0, T, dt, occlusion_scenarios=None):
        from scipy.integrate import solve_ivp
        x0_flat = x0.flatten()
        n = x0_flat.size

        def aug_dyn(t, y):
            x = y[:n].reshape(n, 1)
            Phi = y[n:].reshape((n, n))
            x_dot = self.f_cl(x, occlusion_scenarios, t).flatten()
            Phi_dot = self.F_cl(x, occlusion_scenarios, t) @ Phi
            return np.concatenate([x_dot, Phi_dot.flatten()])

        t_eval = np.arange(0.0, T + 1e-9, dt)
        y0 = np.concatenate([x0_flat, np.eye(n).flatten()])
        sol = solve_ivp(aug_dyn, [0, T], y0, t_eval=t_eval, dense_output=False)
        traj = sol.y[:n, :].T
        Phi_traj = sol.y[n:, :].T.reshape(-1, n, n)
        return traj, Phi_traj, t_eval

    def h_b_stop(self, X):
        # always safe stop set for kinematic unicycle (no velocity state)
        return 1.0

    def grad_h_b_stop(self, X):
        n = X.shape[0]
        grad = np.zeros((1, n))
        return grad
