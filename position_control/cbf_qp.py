import numpy as np
import cvxpy as cp

class CBFQP:
    def __init__(self, robot, robot_spec, num_obs=80):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs

        self.cbf_param = {}

        if self.robot_spec['model'] == "SingleIntegrator2D":
            self.cbf_param['alpha'] = 1.0
        elif self.robot_spec['model'] == 'Unicycle2D':
            self.cbf_param['alpha'] = 1.0
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'DynamicUnicycle2D_C3BF':
            self.cbf_param['alpha'] = 1.0
        elif self.robot_spec['model'] == 'DynamicUnicycle2D_DPCBF':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'DoubleIntegrator2D_DPCBF':
            self.cbf_param['alpha'] = 1.0
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_DPCBF':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_CBFVO':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_OVVO':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_CVAR':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_CNBFVO':
            # This will now be handled by the robot class's gamma parameter
            self.cbf_param['alpha'] = 1.5 
            # === ADDITION: Define activation distances for CBF-VO ===
            self.d_psi_margin = 30.0 
            self.d_v_margin = 35.0
        elif self.robot_spec['model'] == 'Quad2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'Quad3D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5

        self.setup_control_problem()

    def setup_control_problem(self):

        if self.robot_spec['model'] == 'KinematicBicycle2D_CNBFVO':
            # For this model, we need two separate 1D QPs.
            # --- Speed QP (for acceleration u_a) ---
            self.u_a = cp.Variable()
            self.u_a_ref = cp.Parameter()
            self.A_v = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            self.b_v = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            speed_obj = cp.Minimize(cp.sum_squares(self.u_a - self.u_a_ref))
            speed_con = [
                self.A_v @ self.u_a + self.b_v >= 0,
                cp.abs(self.u_a) <= self.robot_spec['a_max']
            ]
            self.speed_problem = cp.Problem(speed_obj, speed_con)

            # --- Steering QP (for steering rate u_r) ---
            self.u_r = cp.Variable()
            self.u_r_ref = cp.Parameter()
            self.A_psi = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            self.b_psi = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            steering_obj = cp.Minimize(cp.sum_squares(self.u_r - self.u_r_ref))
            steering_con = [
                self.A_psi @ self.u_r + self.b_psi >= 0,
                cp.abs(self.u_r) <= self.robot_spec.get('beta_max', np.pi) # Use beta_max or a default
            ]
            self.steering_problem = cp.Problem(steering_obj, steering_con)

        else:
            self.u = cp.Variable((2, 1))
            self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))

            if self.robot_spec['model'] == 'KinematicBicycle2D_OVVO':
                self.num_obs = self.num_obs * 2

            self.A1 = cp.Parameter((self.num_obs, 2), value=np.zeros((self.num_obs, 2)))
            self.b1 = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))

            if self.robot_spec['model'] == 'SingleIntegrator2D':
                constraints = [self.A1 @ self.u + self.b1 >= 0,
                            cp.abs(self.u[0]) <=  self.robot_spec['v_max'],
                            cp.abs(self.u[1]) <=  self.robot_spec['v_max']]
            elif self.robot_spec['model'] == 'Unicycle2D':
                constraints = [self.A1 @ self.u + self.b1 >= 0,
                            cp.abs(self.u[0]) <= self.robot_spec['v_max'],
                            cp.abs(self.u[1]) <= self.robot_spec['w_max']]
            elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DynamicUnicycle2D_C3BF', 'DynamicUnicycle2D_DPCBF']:
                constraints = [self.A1 @ self.u + self.b1 >= 0,
                            cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                            cp.abs(self.u[1]) <= self.robot_spec['w_max']]
            elif self.robot_spec['model'] in ['DoubleIntegrator2D', 'DoubleIntegrator2D_DPCBF']:
                constraints = [self.A1 @ self.u + self.b1 >= 0,
                            cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                            cp.abs(self.u[1]) <= self.robot_spec['a_max']]
            elif self.robot_spec['model'] in ['KinematicBicycle2D', 'KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF', 'KinematicBicycle2D_CBFVO', 'KinematicBicycle2D_OVVO', 'KinematicBicycle2D_CVAR']:
                constraints = [self.A1 @ self.u + self.b1 >= 0,
                            cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                            cp.abs(self.u[1]) <= self.robot_spec['beta_max']]
            elif self.robot_spec['model'] == 'Quad2D':
                constraints = [self.A1 @ self.u + self.b1 >= 0,
                            self.robot_spec["f_min"] <= self.u[0],
                            self.u[0] <= self.robot_spec["f_max"],
                            self.robot_spec["f_min"] <= self.u[1],
                            self.u[1] <= self.robot_spec["f_max"]]
            elif self.robot_spec['model'] == 'Quad3D':
                # overwrite the variables
                self.u = cp.Variable((4, 1))
                self.u_ref = cp.Parameter((4, 1), value=np.zeros((4, 1)))
                self.A1 = cp.Parameter((1, 4), value=np.zeros((1, 4)))
                self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
                objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
                constraints = [self.A1 @ self.u + self.b1 >= 0,
                            self.u[0] <= self.robot_spec['f_max'],
                                self.u[0] >= 0.0,
                            cp.abs(self.u[1]) <= self.robot_spec['phi_dot_max'],
                            cp.abs(self.u[2]) <= self.robot_spec['theta_dot_max'],
                            cp.abs(self.u[3]) <= self.robot_spec['psi_dot_max']]

            self.cbf_controller = cp.Problem(objective, constraints)

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        # 3. Update the CBF constraints for mulit obs
        # if nearest_obs is None:
        #     # deactivate the CBF constraints
        #     self.A1.value = np.zeros_like(self.A1.value)
        #     self.b1.value = np.zeros_like(self.b1.value)
        # elif self.robot_spec['model'] in ['SingleIntegrator2D', 'Unicycle2D', 'KinematicBicycle2D_C3BF']:
        #     h, dh_dx = self.robot.agent_barrier(nearest_obs)
        #     self.A1.value[0,:] = dh_dx @ self.robot.g()
        #     self.b1.value[0,:] = dh_dx @ self.robot.f() + self.cbf_param['alpha'] * h
        # elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D', 'KinematicBicycle2D', 'Quad2D']:
        #     h, h_dot, dh_dot_dx = self.robot.agent_barrier(nearest_obs)
        #     self.A1.value[0,:] = dh_dot_dx @ self.robot.g()
        #     self.b1.value[0,:] = dh_dot_dx @ self.robot.f() + (self.cbf_param['alpha1']+self.cbf_param['alpha2']) * h_dot + self.cbf_param['alpha1']*self.cbf_param['alpha2']*h
        
        robot_radius = self.robot_spec.get('radius', 0.0)
        if self.robot_spec['model'] == 'KinematicBicycle2D_CBFVO':
            u_a_ref_val = control_ref['u_ref'][0, 0]
            u_r_ref_val = control_ref['u_ref'][1, 0]

            # --- STAGE 1: Solve for Safe Acceleration (u_a) ---
            A_v_list, b_v_list = [], []
            active_v_obstacles = 0
            for obs in obs_list:
                d_min_i = robot_radius + obs[2]
                if np.linalg.norm(robot_state[:2, 0] - obs[:2]) < d_min_i + self.d_v_margin:
                    A_v, b_v = self.robot.speed_barrier(robot_state, obs, robot_radius)
                    self.A_v.value[active_v_obstacles, 0] = A_v
                    self.b_v.value[active_v_obstacles, 0] = b_v
                    active_v_obstacles += 1
            
            if active_v_obstacles == 0:
                u_a_safe = u_a_ref_val
            else:
                self.u_a_ref.value = u_a_ref_val
                self.speed_problem.solve(solver=cp.GUROBI, reoptimize=True)
                u_a_safe = self.u_a.value if self.speed_problem.status == 'optimal' else u_a_ref_val

            # --- STAGE 2: Solve for Safe Steering (u_r) ---
            A_psi_list, b_psi_list = [], []
            active_psi_obstacles = 0
            for obs in obs_list:
                d_min_i = robot_radius + obs[2]
                if np.linalg.norm(robot_state[:2, 0] - obs[:2]) < d_min_i + self.d_psi_margin:
                    A_psi, b_psi = self.robot.steering_barrier(robot_state, obs, robot_radius, u_a_safe)
                    self.A_psi.value[active_psi_obstacles, 0] = A_psi
                    self.b_psi.value[active_psi_obstacles, 0] = b_psi
                    active_psi_obstacles += 1
            
            if active_psi_obstacles == 0:
                u_r_safe = u_r_ref_val
            else:
                self.u_r_ref.value = u_r_ref_val
                self.steering_problem.solve(solver=cp.GUROBI, reoptimize=True)
                u_r_safe = self.u_r.value if self.steering_problem.status == 'optimal' else u_r_ref_val
            
            self.status = self.steering_problem.status
            return np.array([[u_a_safe], [u_r_safe]])
        
        else:
            for i in range(min(self.num_obs, len(obs_list))):
                obs = obs_list[i]
                if obs is None:
                    # deactivate the CBF constraints
                    self.A1.value = np.zeros_like(self.A1.value)
                    self.b1.value = np.zeros_like(self.b1.value)
                elif self.robot_spec['model'] in ['SingleIntegrator2D', 'Unicycle2D', 'DynamicUnicycle2D_C3BF', 'DynamicUnicycle2D_DPCBF', 'DoubleIntegrator2D_DPCBF', 'KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF', 'KinematicBicycle2D_CBFVO', 'KinematicBicycle2D_CVAR']:
                    h, dh_dx = self.robot.agent_barrier(obs)
                    self.A1.value[i,:] = dh_dx @ self.robot.g()
                    self.b1.value[i,:] = dh_dx @ self.robot.f() + self.cbf_param['alpha'] * h
                elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D', 'KinematicBicycle2D', 'Quad2D', 'Quad3D']:
                    h, h_dot, dh_dot_dx = self.robot.agent_barrier(obs)
                    self.A1.value[i,:] = dh_dot_dx @ self.robot.g()
                    self.b1.value[i,:] = dh_dot_dx @ self.robot.f() + (self.cbf_param['alpha1']+self.cbf_param['alpha2']) * h_dot + self.cbf_param['alpha1']*self.cbf_param['alpha2']*h
                elif self.robot_spec['model'] == 'KinematicBicycle2D_OVVO':
                    # This now returns two sets of (h, dh_dx)
                    (h_c, dh_dx_c), (h_p, dh_dx_p) = self.robot.agent_barrier(obs)

                    # Constraint 1: Clearance (at row 2*i)
                    self.A1.value[2*i, :] = dh_dx_c @ self.robot.g()
                    self.b1.value[2*i, :] = dh_dx_c @ self.robot.f() + self.cbf_param['alpha'] * h_c

                    # Constraint 2: Passtime (at row 2*i + 1)
                    # We use the same alpha, but you could define a separate one if needed
                    self.A1.value[2*i + 1, :] = dh_dx_p @ self.robot.g()
                    self.b1.value[2*i + 1, :] = dh_dx_p @ self.robot.f() + self.cbf_param['alpha'] * h_p
                    
                solve_val = (self.A1.value[i, :] @ self.u.value + self.b1.value[i, :]
                                if self.u.value is not None else 'N/A')
                # if h > 0 and (solve_val != 'N/A' and solve_val < 0):
                #     print(f"Obstacle {i}: h = {h}, solved constraint value: {solve_val}")
                # print(f"obstacle {i}: dh_dh[0, 3] = {dh_dx[0, 3]} | dh_dh[0, 4] = {dh_dx[0, 4]} | u = {self.u.value}")
            self.u_ref.value = control_ref['u_ref']

            self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
            # print(f'h: {h} | value: {self.A1.value[0,:] @ self.u.value + self.b1.value[0,:]}')

            # Check QP error in tracking.py
            self.status = self.cbf_controller.status

            # print(self.u.value)

            # if self.cbf_controller.status != 'optimal':
            #     raise QPError("CBF-QP optimization failed")

            return self.u.value