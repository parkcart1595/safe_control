""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module implements the PCBF-SF for the nonlinear system"""
import casadi
import numpy as np

from position_control.pcbf_controller import *
from robots.kinematic_bicycle2D_c3bf import KinematicBicycle2D_C3BF
from typing import Dict, Callable
from pytope import Polytope

class SlackOpt():
    """ This class implements the optimal slack computation step"""

    def __init__(self, sys: KinematicBicycle2D_C3BF, X: Polytope, U: Polytope, delta_i: Callable, param_dict: Dict, N: int = 50, use_warm_start: bool = False, verbose: bool = False):
        self.N = N # predict horizon length
        self.verbose = verbose # print debugging
        self.constraint_tol = 0.0
        self.use_warm_start = use_warm_start

        # self.n = sys.state_dim
        # self.m = sys.input_dim
        self.numb_x_constr = X.b.shape[0]

        self.P_f = param_dict['P_f']
        self.gamma_x = param_dict['gamma_x']
        self.gamma_f = param_dict['gamma_f']
        self.alpha_f = param_dict['alpha_f']
        self.T_disc = param_dict['T_disc']

        self.cbf_param = {}
        self.Q = np.diag([50, 50, 1, 1])  # State cost matrix
        self.R = np.array([0.5, 50.0])  # Input cost matrix
        self.cbf_param['alpha'] = 0.2
        self.n_states = 4
            
        self.n_controls = 2
        # For KinematicBicycle2D_C3BF, vs and L (e.g., desired speed, wheelbase) are set here:
        self.vs = 1.0
        self.L = 1.0

        self.opti = casadi.Opti()

        self.x0 = self.opti.parameter(self.n_states, 1) # Initial state
        self.x = self.opti.variable(self.n_states, self.N + 1) # System Dynamics
        self.u = self.opti.variable(self.n_controls, self.N) # Input
        self.gsi = self.opti.variable(X.b.shape[0], self.N) # Slack variable for each step
        self.gsi_N = self.opti.variable() # terminal slack variable (relaxation for terminal constraint)

        # Objective: minimize terminal slack and sum of stage slack norms.
        self.obj = self.alpha_f * self.gsi_N
        for i in range(self.N):
            self.obj += casadi.norm_2(self.gsi[:, i])
        self.opti.minimize(self.obj)

        # Constraints
        self.constraints = [self.x[:, 0] == self.x0]

        for t in range(self.N):
            self.constraints.append(
                self.x[0, t + 1] == self.x[0, t] + self.T_disc * (self.vs + self.x[3, t]) * casadi.sin(self.x[1, t]))
            self.constraints.append(
                self.x[1, t + 1] == self.x[1, t] + self.T_disc * ((self.vs + self.x[3, t]) / self.L) * casadi.tan(self.x[2, t]))
            self.constraints.append(
                self.x[2, t + 1] == self.x[2, t] + self.T_disc * self.u[0, t])
            self.constraints.append(
                self.x[3, t + 1] == self.x[3, t] + self.T_disc * self.u[1, t])

            self.constraints.append(X.A @ self.x[:, t] <= X.b - delta_i(t) * np.ones(X.b.shape) + self.gsi[:,
                                                                                                           t] - self.constraint_tol)  # tightening + relaxation + tolerance
            self.constraints.append(
                U.A @ self.u[:, t] <= U.b - self.constraint_tol)
            self.constraints.append(self.gsi[:, t] >= 0)

        # Terminal constraints
        self.constraints.append(self.gsi_N >= 0)
        self.constraints.append(self.x[:, self.N].T @ self.P_f @ self.x[:,
                                                                        self.N] - self.gamma_x <= self.gsi_N - self.constraint_tol)

        self.opti.subject_to(self.constraints)

        if not self.verbose:
            self.opts = {'ipopt.print_level': 0,
                         'print_time': 0}  # disable output
            self.opti.solver('ipopt', self.opts)
        else:
            self.opti.solver('ipopt')

    def solve(self, x):
        """ Returns the computed optimal slack variables together with the optimal value function.
        In additon, it also returns the optimal input and state trajectory which can be used to warm start
        the next optimziaton problem"""
        self.opti.set_value(self.x0, x.reshape((self.n_states, 1)))

        # Init gsi_temp with nonzero value othw. get ipopt error (Gradients nan etc.)
        if not self.use_warm_start:
            gsi_temp = 1 * np.ones((self.numb_x_constr, self.N))
            self.opti.set_initial(self.gsi, gsi_temp)

        self.sol = self.opti.solve()
        return self.sol.value(self.gsi), self.sol.value(self.gsi_N), self.sol.value(self.obj), self.sol.value(
            self.u), self.sol.value(self.x)


class SafetyFilter():
    """ This class implements the safety filter computation step of the PCBF-SF scheme"""

    def __init__(self, sys: KinematicBicycle2D_C3BF, performance_ctrl: Controller,
                 X: Polytope, U: Polytope, delta_i: Callable, param_dict: Dict, N: int = 50, verbose: bool = False):
        self.N = N
        self.verbose = verbose
        self.constraint_tol = 0.0
        self.perf_ctrl = performance_ctrl

        vs = 1.0
        L = 1.0

        # self.n = sys.state_dim
        # self.m = sys.input_dim
        # u_shape = (self.m, 1)
        # x_shape = (self.n, 1)

        self.P_f = param_dict['P_f']
        self.gamma_x = param_dict['gamma_x']
        self.gamma_f = param_dict['gamma_f']
        self.alpha_f = param_dict['alpha_f']
        self.T_disc = param_dict['T_disc']

        self.cbf_param = {}
        self.Q = np.diag([50, 50, 1, 1])  # State cost matrix
        self.R = np.array([0.5, 50.0])  # Input cost matrix
        self.cbf_param['alpha'] = 0.2
        self.n_states = 4
        self.n_controls = 2

        self.opti = casadi.Opti()

        self.x0 = self.opti.parameter(self.n_states, 1)
        self.ul = self.opti.parameter(self.n_controls, 1) # Provided from performance controller (Could be unsafe)

        self.gsi = self.opti.parameter(X.b.shape[0], self.N)
        self.gsi_N = self.opti.parameter()

        self.x = self.opti.variable(self.n_states, self.N + 1) # Optimized variable for state and input
        self.u = self.opti.variable(self.n_controls, self.N)

        # Objective (Safety Filter)
        self.obj = (self.ul - self.u[:, 0]).T @ (self.ul - self.u[:, 0])

        # Constraints
        self.constraints = [self.x[:, 0] == self.x0] # start point is current x(k)

        for t in range(self.N): # Represent system dynamic using Euler Forward
            self.constraints.append(
                self.x[0, t + 1] == self.x[0, t] + self.T_disc * (vs + self.x[3, t]) * casadi.sin(self.x[1, t]))
            self.constraints.append(
                self.x[1, t + 1] == self.x[1, t] + self.T_disc * ((vs + self.x[3, t]) / L) * casadi.tan(self.x[2, t]))
            self.constraints.append(
                self.x[2, t + 1] == self.x[2, t] + self.T_disc * self.u[0, t])
            self.constraints.append(
                self.x[3, t + 1] == self.x[3, t] + self.T_disc * self.u[1, t])

            self.constraints.append(X.A @ self.x[:, t] <= X.b - delta_i(t) * np.ones(X.b.shape) + self.gsi[:,
                                                                                                           t] - self.constraint_tol)  # tightening + relaxation + tolerance
            self.constraints.append(
                U.A @ self.u[:, t] <= U.b - self.constraint_tol)

        # Terminal constraints
        self.constraints.append(self.x[:, self.N].T @ self.P_f @ self.x[:,
                                                                        self.N] - self.gamma_x <= self.gsi_N - self.constraint_tol)

        self.opti.subject_to(self.constraints)
        self.opti.minimize(self.obj)

        if not self.verbose:
            self.opts = {'ipopt.print_level': 0,
                         'print_time': 0}  # disable output
            self.opti.solver('ipopt', self.opts)
        else:
            self.opti.solver('ipopt')

    def input(self, x, gsi_N, gsi, u_init=None, x_init=None):
        """ Returns a guaranteed safe input which is able to recover states outside the state constraints (but still
        inside the domain of h_PB"""
        ul = self.perf_ctrl.input(x)
        success = False

        # Warm start if available
        if u_init is not None:
            self.opti.set_initial(self.u, u_init)
        if x_init is not None:
            self.opti.set_initial(self.x, x_init)

        self.opti.set_value(self.x0, x.reshape((self.n, 1)))
        self.opti.set_value(self.ul, ul.reshape((self.m, 1)))
        self.opti.set_value(self.gsi, gsi)
        self.opti.set_value(self.gsi_N, gsi_N)

        self.sol = self.opti.solve()
        success = True
        return self.sol.value(self.u)[:, 0], success
    
def compute_collision_cone_cbf(sys: KinematicBicycle2D_C3BF, _x, _u, _obs, robot_radius, beta=1.0):
    """
    Compute the collision cone CBF constraint for the kinematic bicycle 2D model.
    Uses the agent_barrier_dt function defined in KinematicBicycle2D_C3BF.
    
    _x: symbolic state trajectory (from CasADi) [n x ...]
    _u: symbolic input trajectory
    _obs: symbolic obstacle parameter (from TVP)
    robot_radius: scalar, robot's collision radius
    beta: scaling factor (default 1.0)
    """
    # Here we call the agent_barrier_dt method of the robot (sys)
    # Note: agent_barrier_dt expects current state, input, and obstacle information.
    h_val, d_h = sys.agent_barrier_dt(_x, _u, _obs, robot_radius, beta)
    # We form the constraint expression. For example, a typical discrete CBF condition is:
    # Δh(x_k, u_k) + α * h(x_k) ≥ 0, so we rearrange to: - (Δh + α*h) ≤ 0.
    # In our example, we assume _alpha is given in TVP variables.
    if sys.robot_spec['model'] in ['KinematicBicycle2D_C3BF']:
        _alpha = 0.2  # or get from TVP if available.
        cbf_constraint = d_h + _alpha * h_val
    else:
        raise NotImplementedError("Collision cone CBF only implemented for KinematicBicycle2D_C3BF.")
    return cbf_constraint


class PCBF_Algorithm(Controller):
    """ This class combines the slack variables computation step with the safety filter optimziation scheme to get Algorithm 1 """

    def __init__(self, sys: KinematicBicycle2D_C3BF, performance_ctrl: Controller,
                 X: Polytope, U: Polytope, delta_i: Callable, param_dict: Dict, N: int=50, verbose: bool=False):
        self.slack_opt = SlackOpt(
            sys, X, U, delta_i, param_dict, N=N, verbose=verbose)
        self.safety_filter = SafetyFilter(
            sys, performance_ctrl, X, U, delta_i, param_dict, N=N, verbose=verbose)
        self.performance_controller = performance_ctrl
        self.verbose = verbose
        self.hpb_traj = []  # For debugging purposes store h_PB trajectory

    def input(self, x):
        if self.verbose:
            print(f"@x={x}")
        # Step 1 : Compute optimal slack variables
        gsi, gsi_N, hbp, u_init, x_init = self.slack_opt.solve(x)
        self.ustar = u_init
        self.hpb_traj.append(hbp)
        # Step 2 : Compute safety filter problem
        u, success = self.safety_filter.input(x, gsi_N, gsi, u_init, x_init)

        return u

    def reset(self):
        self.hpb_traj = []

if __name__ == "__main__":
    # Dummy implementations for demonstration.
    # Replace these with your actual implementations.
    class DummyKinematicBicycle2D_C3BF(KinematicBicycle2D_C3BF):
        def __init__(self, dt, robot_spec):
            super().__init__(dt, robot_spec)
            self.robot_spec = robot_spec  # store for later use
            self.state_dim = 4
            self.input_dim = 2

        def f_casadi(self, x):
            # Example drift dynamics: for simplicity, use a linear approximation
            # x = [x_pos, y_pos, theta, v]
            A = casadi.DM.eye(4)
            return A @ x  # dummy dynamics

        def g_casadi(self, x):
            return casadi.DM.eye(4,2)  # dummy control influence

        def hf(self, x):
            # Terminal barrier function: x^T*P_f*x - gamma_x, with P_f, gamma_x from robot_spec
            P_f = np.eye(self.state_dim)
            gamma_x = 1.0
            return casadi.mtimes([x.T, P_f, x])[0,0] - gamma_x

        def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.0):
            # Use the provided implementation from KinematicBicycle2D_C3BF.
            # Here we assume obs is given as a CasADi DM vector of appropriate size.
            # For simplicity, we call the original method (you can refine this as needed)
            return self.agent_barrier(x_k, obs, robot_radius, beta)[0], self.agent_barrier(x_k, obs, robot_radius, beta)[1]

    class DummyPerformanceController:
        def __init__(self, state_dim, input_dim):
            self.state_dim = state_dim
            self.input_dim = input_dim

        def input(self, x):
            # For simplicity, return a zero control input (or any nominal feedback)
            return np.zeros((self.input_dim, 1))

    # Create dummy system and performance controller for KinematicBicycle2D_C3BF
    dt = 0.05
    robot_spec = {
        'model': 'KinematicBicycle2D_C3BF',
        'a_max': 0.5,
        'beta_max': 0.5,
        'radius': 0.5
    }
    sys = DummyKinematicBicycle2D_C3BF(dt, robot_spec)
    perf_ctrl = DummyPerformanceController(sys.state_dim, sys.input_dim)

    # Define state and input constraint polytopes (dummy example)
    X = Polytope(np.vstack([np.eye(sys.state_dim), -np.eye(sys.state_dim)]), np.ones((2*sys.state_dim, 1)))
    U = Polytope(np.vstack([np.eye(sys.input_dim), -np.eye(sys.input_dim)]), np.ones((2*sys.input_dim, 1)))

    # Define delta_i function for constraint tightening
    def delta_i(t):
        return t * 0.005

    # Parameter dictionary for PCBF (adjust as needed)
    param_dict = {
        'T_disc': dt,
        'P_f': np.eye(sys.state_dim),
        'gamma_x': 1.0,
        'gamma_f': 1.0,  # not used explicitly here
        'alpha_f': 100.0
    }

    # Create PCBF controller (Predictive CBF algorithm) instance
    pcbf_controller = PCBF_Algorithm(sys, perf_ctrl, X, U, delta_i, param_dict, N=20, verbose=True)

    # Example simulation loop:
    x_current = np.array([1.0, 1.0, 0.0, 0.0])  # initial state (column vector shape (4,1) required)
    for k in range(50):
        u_safe = pcbf_controller.input(x_current)
        print(f"Time step {k}, Safe input: {u_safe}")
        # Update state using dummy dynamics: x(k+1) = x(k) + dt * (f(x) + g(x)*u)
        x_current = x_current + (sys.f_casadi(casadi.DM(x_current)).full().flatten() + 
                                   (sys.g_casadi(casadi.DM(x_current)) @ u_safe).full().flatten()) * dt