import numpy as np
import cvxpy as cp

class NotCompatibleError(Exception):
    '''
    Exception raised for errors when the robot model is not compatible with the controller.
    '''

    def __init__(self, message="Currently not compatible with the robot model. Only compatible with DynamicUnicycle2D now"):
        self.message = message
        super().__init__(self.message)
        
class OptimalDecayCBFQP:
    def __init__(self, robot, robot_spec, num_obs=25):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs

        self.model = self.robot_spec['model']
        self.mode = 'odcbf'

        if self.robot_spec['model'] == 'KinematicBicycle2D_CBFVO':
            self.mode = 'cbfvo'
            self.ku       = 2.0             # range in 0.5 ~ 5.0
            self.kvo      = 80.0           # range in 20 ~ 1000
            self.alpha_vo = 10.0
            self.alpha_c  = 10.0
            # self.cbf_param = {}
            # self.cbf_param['alpha1'] = 0.5
            # self.cbf_param['omega1'] = 1.0  # Initial omega
            # self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
        
        if self.mode == 'odcbf':
            if self.robot_spec['model'] == 'DynamicUnicycle2D': # TODO: not compatible with other robot models yet
                self.cbf_param = {}
                self.cbf_param['alpha1'] = 0.5
                self.cbf_param['alpha2'] = 0.5
                self.cbf_param['omega1'] = 1.0  # Initial omega
                self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
                self.cbf_param['omega2'] = 1.0  # Initial omega
                self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay
            elif self.robot_spec['model'] == 'KinematicBicycle2D':
                self.cbf_param = {}
                self.cbf_param['alpha1'] = 0.5
                self.cbf_param['alpha2'] = 0.5
                self.cbf_param['omega1'] = 1.0  # Initial omega
                self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
                self.cbf_param['omega2'] = 1.0  # Initial omega
                self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay
            else:
                raise NotCompatibleError("Infeasible or Collision")


        self.setup_control_problem()

    def setup_control_problem(self):

        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))

        if self.mode =='odcbf':
            self.omega1 = cp.Variable((1, 1))  # Optimal-decay parameter
            self.omega2 = cp.Variable((1, 1))  # Optimal-decay parameter
            self.A1 = cp.Parameter((1, 2), value=np.zeros((1, 2)))
            self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            self.h = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            self.h_dot = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            
            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref) 
                                    + self.cbf_param['p_sb1'] * cp.square(self.omega1 - self.cbf_param['omega1'])
                                    + self.cbf_param['p_sb2'] * cp.square(self.omega2 - self.cbf_param['omega2']))

            if self.robot_spec['model'] == 'DynamicUnicycle2D':
                constraints = [
                    self.A1 @ self.u + self.b1 + 
                    (self.cbf_param['alpha1'] + self.cbf_param['alpha2'])* self.omega1 @ self.h_dot +
                    self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * self.h @ self.omega2 >= 0,
                    cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                    cp.abs(self.u[1]) <= self.robot_spec['w_max'],
                ]
            elif self.robot_spec['model'] == 'KinematicBicycle2D':
                constraints = [
                    self.A1 @ self.u + self.b1 + 
                    (self.cbf_param['alpha1'] + self.cbf_param['alpha2'])* self.omega1 @ self.h_dot +
                    self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * self.h @ self.omega2 >= 0,
                    cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                    cp.abs(self.u[1]) <= self.robot_spec['beta_max'],
                ]

        if self.mode == 'cbfvo':
            # -------- VO-soft + CBF-hard --------
            # VO slack for each obstacle
            self.lmb = cp.Variable((self.num_obs, 1), nonneg=True)

            # Lfh + Lgh*u >= lambda -> A_vo *u + b_vo - lambda >= 0
            self.A_vo = cp.Parameter((self.num_obs, 2), value=np.zeros((self.num_obs, 2)))
            self.b_vo = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            self.w = cp.Parameter((self.num_obs, 1), nonneg=True, value=np.zeros((self.num_obs, 1))) # 1/TTC

            # Lfh + Lgh*u >= 0 -> A_c *u + b_c >= 0
            self.A_c  = cp.Parameter((self.num_obs, 2), value=np.zeros((self.num_obs, 2)))
            self.b_c  = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))

            J_goal = cp.sum_squares(self.u - self.u_ref)
            J_vo = cp.sum(cp.multiply(self.w, cp.square(self.lmb)))
            objective = cp.Minimize(self.ku * J_goal + self.kvo * J_vo)

            bounds = [cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                        cp.abs(self.u[1]) <= self.robot_spec['beta_max']]
            
            constraints = [
                self.A_vo @ self.u - self.lmb + self.b_vo >= 0,  # VO-soft constraint
                self.A_c  @ self.u + self.b_c  >= 0, # CBF-hard constraint
            ] + bounds

        self.cbf_controller = cp.Problem(objective, constraints)

    @staticmethod
    def ttc_circle(p_rel, v_rel, R, eps=1e-9):
        # Solve ||p + t v|| = R for t>0
        a = float(v_rel.T @ v_rel)
        b = 2.0*float(p_rel.T @ v_rel)
        c = float(p_rel.T @ p_rel) - R**2
        if a < eps:
            return np.inf
        disc = b*b - 4*a*c
        if disc <= 0:
            return np.inf
        t1 = (-b - np.sqrt(disc)) / (2*a)
        t2 = (-b + np.sqrt(disc)) / (2*a)
        ts = [t for t in (t1, t2) if t > 0]
        return min(ts) if ts else np.inf

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        
        X = robot_state
        self.u_ref.value = control_ref['u_ref']

        if self.mode == 'odcbf':
            # Update the CBF constraints
            if nearest_obs is None:
                self.A1.value = np.zeros_like(self.A1.value)
                self.b1.value = np.zeros_like(self.b1.value)
                self.h.value = np.zeros_like(self.h.value)
                self.h_dot.value = np.zeros_like(self.h_dot.value)
            else:
                h, h_dot, dh_dot_dx = self.robot.agent_barrier(nearest_obs)
                self.A1.value[0,:] = dh_dot_dx @ self.robot.g()
                self.b1.value[0,:] = dh_dot_dx @ self.robot.f()
                self.h.value[0,:] = h
                self.h_dot.value[0,:] = h_dot

        if self.mode == 'cbfvo':
            if obs_list is None:
                obs_list = []
            # if not isinstance(obs_list, (list, tuple)):
            #     obs_list = [obs_list]

            m = min(self.num_obs, len(obs_list))
            A_vo = np.zeros((self.num_obs, 2))
            b_vo = np.zeros((self.num_obs, 1))
            A_c  = np.zeros((self.num_obs, 2))
            b_c  = np.zeros((self.num_obs, 1))
            wvec = np.zeros((self.num_obs, 1))

            f = self.robot.f()  # (4x1)
            g = self.robot.g()  # (4x2)

            theta = X[2,0]; v = X[3,0]

            for i in range(m):
                obs = obs_list[i]
                if obs is None:
                    continue

                # --- VO guidance (soft) ---
                h_vo, dhdx_vo = self.robot.agent_barrier_vo(obs)
                Lf_vo = float(dhdx_vo @ f)
                Lg_vo = (dhdx_vo @ g).reshape(-1)  # (2,)
                A_vo[i,:] = Lg_vo
                b_vo[i,0] = Lf_vo + self.alpha_vo * h_vo

                # --- Hard safety CBF ---
                h_c, dhdx_c = self.robot.agent_barrier(obs)
                Lf_c = float(dhdx_c @ f)
                Lg_c = (dhdx_c @ g).reshape(-1)

                ovx, ovy = (obs[3], obs[4]) if len(obs) > 3 else (0.0, 0.0)
                p_rel = np.array([[obs[0]-X[0,0]],
                                  [obs[1]-X[1,0]]])
                rho = max(np.linalg.norm(p_rel), 1e-8)
                p_hat = p_rel / rho
                v_rel = np.array([[ovx - v*np.cos(theta)],
                                  [ovy - v*np.sin(theta)]])
                nu = float(v_rel.T @ p_hat)

                if nu >= 0.0:
                    # deactivate
                    A_c[i,:] = 0.0
                    b_c[i,0] = 0.0
                else:
                    A_c[i,:] = Lg_c
                    b_c[i,0] = Lf_c + self.alpha_c * h_c

                # --- VO weight: 1/TTC ---
                R = float(obs[2] + self.robot_spec['radius'])
                ttc = self.ttc_circle(p_rel, v_rel, R)
                wvec[i,0] = 0.0 if np.isinf(ttc) else 1.0/max(ttc, 1e-3)
                wvec = np.maximum(wvec, 0.0)
            # set params
            self.A_vo.value = A_vo
            self.b_vo.value = b_vo
            self.A_c.value  = A_c
            self.b_c.value  = b_c
            self.w.value    = wvec


        # print(self.omega1.value, self.omega2.value)

        # self.u_ref.value = control_ref['u_ref']

        # Solve the optimization problem
        self.cbf_controller.solve(solver=cp.GUROBI)
        self.status = self.cbf_controller.status
        
        return self.u.value