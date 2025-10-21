import numpy as np
import cvxpy as cp

class BackupCBFQP:
    def __init__(self, robot, robot_spec, num_obs=10):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs

        # Backup CBF parameters
        self.T_horizon = 1.5   # backup time T
        self.dt_backup = 0.1   # backup trajectory discrete step
        self.alpha = 1.0       # Class-K function

        self.setup_control_problem()

    def setup_control_problem(self):
        # QP variables and parameters
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))

        max_constraints = 5 * int(self.T_horizon / self.dt_backup + 2)
        self.A_cbf = cp.Parameter((max_constraints, 2), value=np.zeros((max_constraints, 2)))
        self.b_cbf = cp.Parameter((max_constraints, 1), value=np.zeros((max_constraints, 1)))
        
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
        
        constraints = [
            self.A_cbf @ self.u <= self.b_cbf,
            cp.abs(self.u[0]) <= self.robot_spec['a_max'],
            cp.abs(self.u[1]) <= self.robot_spec['a_max']
        ]

        self.cbf_controller = cp.Problem(objective, constraints)
        self.status = None

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        self.u_ref.value = control_ref['u_ref']
        
        if obs_list is None:
            # if no obstacle, use u_ref
            self.status = 'optimal'
            return self.u_ref.value

        A_list, b_list = [], []
        
        phi_b, Phi_b, tau_points = self.robot.simulate_backup_trajectory(
            robot_state, self.T_horizon, self.dt_backup
        )
        
        # 1. obsatcle avoidance constraints (trajectory constraint)
        for obs in obs_list:

            for i in range(1, len(tau_points)): # except tau=0
                phi_i = phi_b[i].reshape(-1, 1)
                Phi_i = Phi_b[i]

                # h(x) = ||x_pos - obs_pos||^2 - d_min^2
                d_min = obs[2] + self.robot_spec['radius']
                h_i = np.linalg.norm(phi_i[0:2] - obs[0:2])**2 - d_min**2
                
                pos_diff_col = phi_i[0:2] - obs[0:2].reshape(-1, 1) # (2,1) - (2,1) = (2,1)
                grad_h_i = 2 * np.hstack([pos_diff_col.T, np.array([[0, 0]])])

                # calculate Lie Derivative)
                # h_dot = grad_h @ (Phi @ (f(x) + g(x)u))
                Lfh_i = grad_h_i @ Phi_i @ self.robot.f(robot_state)
                Lgh_i = grad_h_i @ Phi_i @ self.robot.g(robot_state)

                # QP: A*u <= b
                A_list.append(-Lgh_i)
                b_list.append(Lfh_i + self.alpha * h_i)

        # 2. Backup set constraint (final state constraint)
        # final state phi_T should reach backup set
        phi_T = phi_b[-1].reshape(-1, 1)
        Phi_T = Phi_b[-1]
        
        h_b_T = self.robot.h_b_stop(phi_T)
        grad_h_b_T = self.robot.grad_h_b_stop(phi_T)

        Lfh_b_T = grad_h_b_T @ Phi_T @ self.robot.f(robot_state)
        Lgh_b_T = grad_h_b_T @ Phi_T @ self.robot.g(robot_state)
        A_list.append(-Lgh_b_T)
        b_list.append(Lfh_b_T + self.alpha * h_b_T)

        # 3. QP parameter update and solve
        num_constraints = len(A_list)
        A_cbf_val = np.array(A_list).reshape(num_constraints, 2)
        b_cbf_val = np.array(b_list).reshape(num_constraints, 1)

        self.A_cbf.value[:, :] = 0
        self.A_cbf.value[:num_constraints, :] = A_cbf_val
        self.b_cbf.value[:, :] = 1e6
        self.b_cbf.value[:num_constraints, :] = b_cbf_val
        
        self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
        self.status = self.cbf_controller.status
        
        if self.status != 'optimal':
            # if qp is not solved, use backup policy
            return self.robot.backup_input(robot_state)
            
        return self.u.value