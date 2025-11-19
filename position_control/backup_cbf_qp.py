import numpy as np
import cvxpy as cp
import time
from matplotlib.path import Path

class InfeasibleError(Exception):
    '''
    Exception raised for errors when QP is infeasible or 
    the robot collides with the obstacle
    '''

    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)

class BackupCBFQP:
    def __init__(self, robot, robot_spec, num_obs=10, kappa=10.0):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs
        self.kappa = kappa
        self.occlusion_scenarios = []
        
        self.sensing_range = float(self.robot_spec.get('sensing_range', 10.0))
        self.debug = bool(self.robot_spec.get('debug_backup_qp', False))

        # Backup CBF parameters
        self.T_horizon = 3.0   # backup time T
        self.dt_backup = 0.05   # backup trajectory sampling time step
        self.alpha = 1.0       # Class-K function

        self.setup_control_problem()
        
        target = self.robot
        try:
            if hasattr(target, "set_occ_barrier_fn"):
                target.set_occ_barrier_fn(self._occlusion_barrier_softmax_curved)
            elif hasattr(target, "robot") and hasattr(target.robot, "set_occ_barrier_fn"):
                target.robot.set_occ_barrier_fn(self._occlusion_barrier_softmax_curved)
            else:
                print("[BackupCBFQP][WARN] Could not inject occlusion barrier callback "
                    "(no set_occ_barrier_fn on robot).")
        except Exception as e:
            print(f"[BackupCBFQP][WARN] Failed to inject occ barrier callback: {e}")
        
    def _occlusion_barrier_softmax(self, pos, scenario, tau):
        """
        Compute soft-max occlusion barrier and its gradient at a backup state.

        Parameters
        ----------
        scenario : dict
            Contains:
                'A'         : (M, 2) half-space normals
                'b0'        : (M,) initial offsets
                'v_adv_max' : float, adversary speed bound
        tau : float
            Look-ahead time along the backup trajectory.

        Returns
        -------
        h_tilde : float or None
            Soft-max barrier value. None if invalid.
        grad_pos : np.ndarray or None, shape (1, 2)
            Gradient of h_tilde w.r.t. position. None if invalid.
        """
        A = scenario['A']      # (M,2)
        b0 = scenario['b0']    # (M,)
        v_adv = scenario['v_adv_max']
        R = self.robot_spec['radius']
        kappa = self.kappa

        if A.size == 0 or b0.size == 0:
            return None, None
        
        # b_i(τ) = b0_i + v_adv * tau
        b_tau = b0 + v_adv * tau  # (M,)

        # h_i = a_i^T pos - b_i(τ) - R
        # pos: (2,1) -> (M,)
        pos_flat = pos.reshape(2,)
        h_i = A @ pos_flat - b_tau - R   # (M,)
        
        if not np.all(np.isfinite(h_i)):
            return None, None

        M = h_i.shape[0]
        if M == 0:
            return None, None

        # Numerically stable log-sum-exp soft-max over {h_i}
        max_hi = np.max(h_i)
        z = np.exp(kappa * (h_i - max_hi))
        Z = np.sum(z)

        if not np.isfinite(Z) or Z <= 0.0:
            return None, None
        
        lse = max_hi + np.log(Z)
        h_tilde = (lse - np.log(M)) / kappa
        
        # softmax weights
        # w = np.exp(kappa * h_i)
        # w /= np.sum(w)
        # exp_shifted = np.exp(kappa * (h_i - max_hi))
        # w = exp_shifted / np.sum(exp_shifted)

        w = z / Z  # (M,)
        if not np.all(np.isfinite(w)):
            return None, None
        
        # grad wrt pos: sum_i w_i * a_i
        grad_pos = (w[:, None] * A).sum(axis=0, keepdims=True)  # (1,2)
        
        if not np.all(np.isfinite(grad_pos)):
            return None, None

        return float(h_tilde), grad_pos  # grad_pos (1,2)
    
    def _occlusion_barrier_softmax_curved(self, pos, scenario, tau=0.0):
        """
        Curved occlusion barrier:
        - h1, h2 : two tangent half-space
        - h3     : sensing range arc
        - h4     : obstacle arc

        h_tilde(x) = (1/kappa) [ log Σ exp(kappa h_i(x)) - log(4) ]
        """
        p = scenario['robot_pos']       # robot position
        c = scenario['obs_center']      # obstacle center
        R_o = scenario['obs_radius']    # obstacle radius
        pi_adv = scenario['v_adv_max']
        R = self.robot_spec['radius']   # robot radius (margin)
        kappa = self.kappa
        
        A = scenario.get('A', None)
        b0 = scenario.get('b0', None)
        if A is None or b0 is None or A.shape[0] < 2:
            return None, None, None
        
        pos = np.asarray(pos, float).reshape(2,)
        x, y = pos

        # --- tangent halfspaces (use first 2 rows) ---
        a1, a2 = A[3], A[1]
        beta1, beta2 = b0[3], b0[1]

        # safe if outside wedge (plus robot radius margin)
        h1 = a1 @ pos - beta1 - R
        h2 = a2 @ pos - beta2 - R
        
        A_stack = []
        if h1 >=0:
            A_stack.append(a1)
        if h2 >=0:
            A_stack.append(a2)
            
        if A_stack:
            risk_normal_vec = np.vstack(A_stack)
        else:
            risk_normal_vec = np.empty((0,2))

        # --- expanded obstacle disk ---
        R_occ = pi_adv * float(tau)
        dx_o = x - c[0]
        dy_o = y - c[1]
        # safe if outside expanded disk (plus robot radius)
        h3 = np.sqrt(dx_o*dx_o + dy_o*dy_o - (R_occ + R + R_o)**2)

        h_i = np.array([h1, h2, h3], dtype=float)
        # print(h_i)
        if not np.all(np.isfinite(h_i)):
            return None, None, None

        # log-sum-exp soft-max
        M = h_i.size
        max_h = np.max(h_i)
        # print(max_h)
        z = np.exp(kappa * (h_i - max_h))
        Z = np.sum(z)
        if not np.isfinite(Z) or Z <= 0.0:
            return None, None, None

        lse = max_h + np.log(Z)
        h_tilde = (lse - np.log(M)) / kappa
        # print(h_tilde)
        # gradient:
        # ∂h1/∂x = a1
        # ∂h2/∂x = a2
        # ∂h3/∂x = 2(x-p)
        # ∂h4/∂x = -2(x-c)
        dh1 = a1
        dh2 = a2
        # dh3 = np.array([2.0 * dx_s, 2.0 * dy_s])
        # dh3 = 2.0 * np.array([dx_o, dy_o])
        eps = 1e-9
        dh3 = (1.0 / (h3 + eps)) * np.array([dx_o, dy_o])
        

        grads = np.vstack([dh1, dh2, dh3])  # (4,2)

        lambda_i = z / Z
        grad_pos = (lambda_i[:, None] * grads).sum(axis=0, keepdims=True)

        if not np.all(np.isfinite(grad_pos)):
            return None, None, None

        return float(h_tilde), grad_pos, risk_normal_vec

    
    def _circle_tangents(self, p, c, R):
        """
        Compute tangent points from point p to a circle centered at c with radius R.

        t1, t2 :
            Tangent points on the circle (each (2,))
            if p is inside/on the circle (no valid tangents).
        """
        p = np.asarray(p, dtype=float).reshape(2,)
        c = np.asarray(c, dtype=float).reshape(2,)
        v = p - c
        d2 = float(v @ v)
        R2 = R * R

        # No tangent if p is inside or on the circle
        if d2 <= R2:
            return None, None

        x1, y1 = v
        x0 = R2 * x1 / d2
        y0 = R2 * y1 / d2
        k = R * np.sqrt(d2 - R2) / d2

        # Tangent points in global coordinates
        t1 = np.array([x0 - y1 * k, y0 + x1 * k]) + c
        t2 = np.array([x0 + y1 * k, y0 - x1 * k]) + c
        return t1, t2
    
    def _segment_intersects_circle(self, p, x, c, R):
        """
        Check if segment [p, x] intersects the circle centered at c with radius R.
        Used to define 'true' occlusion: line-of-sight blocked by obstacle.
        """
        p = np.asarray(p, float)
        x = np.asarray(x, float)
        c = np.asarray(c, float)

        d = x - p
        f = p - c

        a = d @ d
        b = 2.0 * (f @ d)
        c_term = f @ f - R**2

        disc = b*b - 4.0*a*c_term
        if disc < 0.0 or a <= 1e-12:
            return False

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # intersection lies on the segment if 0 <= t <= 1
        return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)

    
    def _polygon_to_halfspaces(self, poly):
        """
        Convert a convex polygon into half-space form: { z | A z <= b }.
        """
        poly = np.asarray(poly, dtype=float)
        M = poly.shape[0]
        if M < 3:
            return None, None

        centroid = np.mean(poly, axis=0)

        A_list = []
        b_list = []

        for i in range(M):
            p1 = poly[i]
            p2 = poly[(i + 1) % M]
            edge = p2 - p1
            if np.linalg.norm(edge) < 1e-9:
                continue

             # Outward normal candidate (right-hand normal)
            n = np.array([edge[1], -edge[0]], dtype=float)
            n /= np.linalg.norm(n)
            
            b = float(n @ p1)

            # Ensure centroid is inside: n^T centroid <= b
            if n @ centroid > b + 1e-9:
                n = -n
                b = -b

            A_list.append(n)
            b_list.append(b)
            
        if len(A_list) == 0:
            return None, None

        A = np.vstack(A_list)        # (M_eff, 2)
        b0 = np.array(b_list)        # (M_eff,)
        
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b0))):
            return None, None
        
        return A, b0
    
    def _build_occlusion_scenario(self, robot_state, obs):
        """
        Build an occlusion scenario for a single circular obstacle.

        The occlusion region is a wedge from the robot through the tangent points to a far range, then converted to half-spaces.

        scenario :
            {
              'A'         : (M_k, 2) half-space normals
              'b0'        : (M_k,) offsets
              'v_adv_max' : float, adversary speed bound
              'poly'      : (4, 2) occlusion polygon vertices
            }
            Returns None if no valid occlusion is formed.
        """

        px = float(robot_state[0, 0])
        py = float(robot_state[1, 0])
        p = np.array([px, py])

        obs = np.asarray(obs).flatten()
        ox, oy, r_obs = obs[:3]
        c = np.array([ox, oy])
        R_o = float(r_obs)

        sensing_R = self.sensing_range
        v_adv = float(self.robot_spec.get('v_adv_max_occ', 0.5))
        
        p_rel = np.array([[px - ox], 
                        [py - oy]])
        
        p_rel_mag = np.linalg.norm(p_rel)
        arc_adv = v_adv * p_rel / p_rel_mag
        arc_adv = arc_adv.flatten()

        # Ignore obstacle if it is outside sensing range
        d = np.linalg.norm(c - p)
        if d >= sensing_R:
            return None

        # Compute tangent points from robot to obstacle
        t1, t2 = self._circle_tangents(p, c, R_o)
        if t1 is None or t2 is None:
            return None

        # Extend tangent directions to sensing range to form occlusion wedge
        dir1 = t1 - p
        n1 = np.linalg.norm(dir1)
        if n1 < 1e-6:
            return None
        dir1 /= n1  # tangent 1 unit vector
        
        dir2 = t2 - p
        n2 = np.linalg.norm(dir2)
        if n2 < 1e-6:
            return None
        dir2 /= n2  # tangenet 2 unit vector

        far1 = p + sensing_R * dir1
        far2 = p + sensing_R * dir2

        # occlusion polygon: [t1, t2, far2, far1]
        poly = np.vstack([t1, t2, far2, far1])

        A, b0 = self._polygon_to_halfspaces(poly)
        if A is None:
            return None
        
        scenario = {
            'A': A,
            'b0': b0,
            'v_adv_max': v_adv,
            'arc_adv': arc_adv,
            'poly': poly,
            ## For arc softmax
            'robot_pos': p,
            'obs_center': c,
            'obs_radius': R_o,
            't1': t1,
            't2': t2,
        }
        
        _, _, risk_vec = self._occlusion_barrier_softmax_curved(p, scenario, tau=0.0)
        
        risk_vec *= v_adv
        scenario['risk_normal_vec']= risk_vec
        
        return scenario
    
    def _point_in_poly(self, pt, poly):
        # poly: (N,2), pt: (2,)
        path = Path(poly)
        return path.contains_point((float(pt[0]), float(pt[1])), radius=1e-12)
        
    def _filter_visible_and_build_occ(self, robot_state, obs_list):
        
        visible_obs = []
        occl_scenarios = []
        wedge_paths = []

        if obs_list is None:
            return visible_obs, occl_scenarios

        obs_arr = np.array(obs_list, dtype=float)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)

        px, py = float(robot_state[0, 0]), float(robot_state[1, 0])
        p = np.array([px, py])
        R_sense2 = self.sensing_range ** 2

        keep = []
        for k, o in enumerate(obs_arr):
            if (o[0]-px)**2 + (o[1]-py)**2 <= R_sense2:
                keep.append(k)
        if not keep:
            return visible_obs, occl_scenarios

        obs_arr = obs_arr[keep]

        dists = np.linalg.norm(obs_arr[:, :2] - p[None, :], axis=1)
        order = np.argsort(dists)

        for idx in order:
            obs = obs_arr[idx]
            c = obs[:2]

            occluded = any(self._point_in_poly(c, sc['poly']) for sc in occl_scenarios)
            if occluded:
                continue

            visible_obs.append(obs)

            sc = self._build_occlusion_scenario(robot_state, obs)
            if sc is not None and sc.get('poly') is not None:
                occl_scenarios.append(sc)

        return visible_obs, occl_scenarios

    def visualize_occlusion_curved_debug(self, robot_state, obs, kappa=None, grid_res=0.05, tau=0.0):
        """
        Debug visualization for ONE obstacle:
          - True occlusion region (LOS-blocked, using circle arcs implicitly)
          - Curved softmax occlusion set { x | h_tilde_curved(x) <= 0 }
          - Polygon wedge used to build (A, b0)

        """
        import matplotlib.pyplot as plt

        if kappa is not None:
            old_kappa = self.kappa
            self.kappa = kappa
        else:
            old_kappa = self.kappa

        # Robot state & obstacle
        px = float(robot_state[0, 0])
        py = float(robot_state[1, 0])
        p = np.array([px, py])

        obs = np.asarray(obs, dtype=float).flatten()
        ox, oy, r_obs = obs[:3]
        c = np.array([ox, oy])
        R_obs = float(r_obs)

        sensing_R = self.sensing_range

        # Build scenario (includes robot_pos, obs_center, etc.)
        scenario = self._build_occlusion_scenario(robot_state, obs)
        if scenario is None:
            print("[viz] No valid occlusion scenario.")
            self.kappa = old_kappa
            return

        poly = scenario['poly']

        # ----- Grid -----
        margin = sensing_R + 0.5
        xmin = px - margin
        xmax = px + margin
        ymin = py - margin
        ymax = py + margin

        xs = np.arange(xmin, xmax + grid_res, grid_res)
        ys = np.arange(ymin, ymax + grid_res, grid_res)
        XX, YY = np.meshgrid(xs, ys)

        true_occ = np.zeros_like(XX, dtype=bool)
        h_curved = np.full_like(XX, np.nan, dtype=float)

        # ----- 1) True occlusion region (LOS block) -----
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                x = np.array([XX[i, j], YY[i, j]])

                if np.linalg.norm(x - p) > sensing_R:
                    continue

                if self._segment_intersects_circle(p, x, c, R_obs):
                    true_occ[i, j] = True

        # ----- 2) Curved softmax barrier field h_tilde_curved(x) -----
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                x = np.array([XX[i, j], YY[i, j]])
                pos_col = x.reshape(2, 1)

                h_tilde, _, _ = self._occlusion_barrier_softmax_curved(
                    pos_col,
                    scenario,
                    tau=tau
                )
                if h_tilde is not None and np.isfinite(h_tilde):
                    h_curved[i, j] = h_tilde

        # ----- Plot -----
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')

        # Robot
        ax.plot(px, py, 'ko', markersize=5, label='robot')

        # Sensing circle
        sensing_circle = plt.Circle((px, py), sensing_R,
                                    fill=False, linestyle='--', color='gray')
        ax.add_patch(sensing_circle)

        # Obstacle
        obs_circle = plt.Circle((ox, oy), R_obs,
                                fill=False, color='k')
        ax.add_patch(obs_circle)

        # True occlusion (LOS-based)
        tc = ax.contourf(
            XX, YY, true_occ,
            levels=[0.5, 1.5],
            alpha=0.25,
            colors=['#ffcccc']
        )
        tc.collections[0].set_label('true occlusion (LOS)')

        # Curved softmax 0-level set (QP에서 사용하는 approx boundary)
        # h_tilde(x) = 0 레벨셋을 contour로 그림
        try:
            cs = ax.contour(
                XX, YY, h_curved,
                levels=[0.0],
                colors='red',
                linewidths=2.0
            )
            for cset in cs.collections:
                cset.set_label('curved softmax h̃(x)=0')
        except Exception:
            print("[viz] Warning: could not draw curved softmax contour (maybe all NaN).")

        # Polygon wedge
        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1],
                'b--', linewidth=1.5, label='polygon wedge')

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title("True occlusion vs curved softmax approx vs polygon wedge")

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc='upper right')

        plt.show()

        # restore kappa
        self.kappa = old_kappa

    # BackupCBFQP class 내부 (도우미 추가)
    def _u_pi_at(self, x, scenarios):
        # 1) 로봇이 백업-at 인터페이스를 주면 사용
        if hasattr(self.robot, "backup_input_at"):
            return self.robot.backup_input_at(x, scenarios)
        # 2) 시야차단 aware 백업
        if (scenarios is not None) and hasattr(self.robot, "backup_input_occlusion"):
            u = self.robot.backup_input_occlusion(x, scenarios)
            if u is not None:
                return u
        # 3) 일반 백업
        if hasattr(self.robot, "backup_input"):
            u = self.robot.backup_input(x)
            if u is not None:
                return u
        # 4) 끝수단
        if hasattr(self.robot, "stop"):
            return self.robot.stop(x)
        return np.zeros((2,1), dtype=float)

    
    def setup_control_problem(self):
        # QP variables and parameters
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))

        # max_constraints = 5 * int(self.T_horizon / self.dt_backup + 2)
        
        N_tau = int(self.T_horizon / self.dt_backup) + 2
        max_constraints = int((2 * self.num_obs + 10) * N_tau)
        
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
        
    def set_occlusion_scenarios(self, scenarios):
        """
        Set occlusion-based adversarial scenarios.
        """
        self.occlusion_scenarios = scenarios
        
    def solve_control_problem(self, robot_state, control_ref, obs_list):
        self.u_ref.value = control_ref['u_ref']

        ######### Only detected obs version ########
        # if obs_list is None:
        #     self.status = 'optimal'
        #     return self.u_ref.value
        
        # if isinstance(obs_list, np.ndarray):
        #     if obs_list.size == 0:
        #         self.status = 'optimal'
        #         return self.u_ref.value

        # elif isinstance(obs_list, (list, tuple)):
        #     if len(obs_list) == 0:
        #         self.status = 'optimal'
        #         return self.u_ref.value
        
        # no obstacle and no occlusion => nominal
        # no_obs = (
        #     (obs_list is None) or
        #     (isinstance(obs_list, np.ndarray) and obs_list.size == 0) or
        #     (isinstance(obs_list, (list, tuple)) and len(obs_list) == 0)
        # )
        
        # 0) Filter obstacles within sensing range
        detected_obs = []
        
        if obs_list is not None:
            obs_arr = np.array(obs_list, dtype=float)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)

            px, py = float(robot_state[0, 0]), float(robot_state[1, 0])
            R_sense2 = self.sensing_range **2
            p = np.array([px, py])
            
            for obs in obs_arr:
                ox = float(obs[0])
                oy = float(obs[1])
                if (ox - px) ** 2 + (oy - py) ** 2 <= R_sense2:
                    detected_obs.append(obs)

        no_obs = (len(detected_obs) == 0)

        # 1) Build occlusion scenarios from detected obstacles
        # occlusion_scenarios = []
        # if not no_obs:
        #     for obs in detected_obs:
        #         scenario = self._build_occlusion_scenario(robot_state, obs)
        #         if scenario is not None:
        #             occlusion_scenarios.append(scenario)
        # self.occlusion_scenarios = occlusion_scenarios
        # no_occ = (len(self.occlusion_scenarios) == 0)
        
        visible_obs, occlusion_scenarios = self._filter_visible_and_build_occ(robot_state, obs_list)
        self.occlusion_scenarios = occlusion_scenarios

        no_obs = (len(visible_obs) == 0)
        no_occ = (len(self.occlusion_scenarios) == 0)

        # 2) If there is no obstacle and no occlusion, use nominal control
        if no_obs and no_occ:
            self.status = 'optimal'
            if self.debug:
                print("[BackupCBFQP] no detected obstacle/occlusion -> use u_ref")
            return self.u_ref.value

        A_list, b_list = [], []

        # 3) Compute backup trajectory under pi_backup
        phi_b, Phi_b, tau_points = self.robot.simulate_backup_trajectory(
            robot_state, self.T_horizon, self.dt_backup,
            occlusion_scenarios=None if no_occ else self.occlusion_scenarios
        )

        # Pre-compute f(x), g(x) at current state for Lie derivatives
        f_x = self.robot.f(robot_state)   # (4,1)
        g_x = self.robot.g(robot_state)   # (4,2)
        
        # 4) Safety constraints for detected obstacles (instantaneous HOCBF)
        if not no_obs:
            x = float(robot_state[0, 0])
            y = float(robot_state[1, 0])
            vx = float(robot_state[2, 0])
            vy = float(robot_state[3, 0])
            p = np.array([x, y])
            v = np.array([vx, vy])

            R_robot = self.robot_spec['radius']
            gamma1 = 1.0  # class k_1
            gamma2 = 1.0  # class k_2

            for obs in visible_obs:
                obs = np.asarray(obs, dtype=float).flatten()
                ox, oy, r_obs = obs[:3]
                if len(obs) >= 5:
                    vx_o, vy_o = obs[3:5]
                else:
                    vx_o, vy_o = 0.0, 0.0

                p_obs = np.array([ox, oy])
                v_obs = np.array([vx_o, vy_o])

                p_rel = p - p_obs
                v_rel = v - v_obs

                d_min = r_obs + R_robot

                h = float(p_rel @ p_rel - d_min**2)
                # relative degree 2 HOCBF
                h_dot = 2.0 * float(p_rel @ v_rel)

                # h_ddot = 2||v_rel||^2 + 2 p_rel^T u
                v_rel_norm2 = float(v_rel @ v_rel)

                psi1 = h_dot + gamma1 * h

                A = -2.0 * p_rel.reshape(1, 2)  # (1,2)
                b = 2.0 * v_rel_norm2 + gamma2 * psi1  # scalar

                if np.all(np.isfinite(A)) and np.isfinite(b):
                    A_list.append(A)
                    b_list.append(np.array([[b]]))
                    
            # for obs in obs_list:
            #     obs = np.asarray(obs).flatten()
            #     ox, oy, r_obs = obs[:3]

            #     if len(obs) >= 5:
            #         vx_o, vy_o = obs[3:5]
            #     else:
            #         vx_o, vy_o = 0.0, 0.0

            #     for i in range(1, len(tau_points)):  # tau = 0 제외
            #         tau = tau_points[i]

            #         phi_i = phi_b[i].reshape(-1, 1)   # (4,1)
            #         Phi_i = Phi_b[i]                  # (4,4)

            #         obs_pos_tau = np.array([
            #             [ox + vx_o * tau],
            #             [oy + vy_o * tau]
            #         ])

            #         # h^c(x, t+tau) = ||p(τ) - p_obs(τ)||^2 - d_min^2
            #         d_min = r_obs + self.robot_spec['radius']
            #         diff = phi_i[0:2] - obs_pos_tau            # (2,1)
            #         h_i = float(diff.T @ diff - d_min**2)

            #         # ∂h/∂phi = [2 (p - p_obs(τ))^T, 0, 0]
            #         grad_h_phi = 2.0 * np.hstack([diff.T, np.array([[0.0, 0.0]])])  # (1,4)

            #         # backup CBF chain rule
            #         f_x = self.robot.f(robot_state)      # (4,1)
            #         g_x = self.robot.g(robot_state)      # (4,2)

            #         Lfh_i = grad_h_phi @ Phi_i @ f_x     # (1,1)
            #         Lgh_i = grad_h_phi @ Phi_i @ g_x     # (1,2)

            #         # CBF inequality: L_f h + L_g h u + α h ≥ 0
            #         A_list.append(-Lgh_i)
            #         b_list.append(Lfh_i + self.alpha * h_i)
                    
        # 5) Occlusion-based adversarial constraints (backup CBF along trajectory)
        if not no_occ:
            for scenario in self.occlusion_scenarios:
                for i in range(1, len(tau_points)):
                    tau = tau_points[i]
                    phi_i = phi_b[i].reshape(-1, 1)
                    Phi_i = Phi_b[i]

                    pos_i = phi_i[0:2]  # (2,1)
                    # print(f"pos_i : {pos_i}")

                    h_tilde, grad_pos, _ = self._occlusion_barrier_softmax_curved(
                        pos_i, scenario, tau
                    )
                    # if h_tilde is None:
                    #     continue
                        
                    # if h_tilde < 0.0:
                    #     if self.debug:
                    #         print(f"[occ] skip at tau={tau:.2f}, h_tilde={h_tilde:.3e}<0")
                    #     continue
                    
                    if h_tilde is None or grad_pos is None:
                        continue

                    # Gradient w.r.t. backup state φ: [grad_pos, 0, 0]
                    grad_h_phi = np.hstack(
                        [grad_pos, np.array([[0.0, 0.0]])]
                    )  # (1,4)

                    # f_x = self.robot.f(robot_state)   # (4,1)
                    # g_x = self.robot.g(robot_state)   # (4,2)
                    
                    u_pi_phi = self._u_pi_at(phi_i, self.occlusion_scenarios)
                    u_pi_phi = np.asarray(u_pi_phi, dtype=float).reshape(2,1)
                    f_pi_phi = self.robot.f(phi_i) + self.robot.g(phi_i) @ u_pi_phi

                    Lfh = grad_h_phi @ (Phi_i @ f_x - f_pi_phi)    # (1,1)
                    Lgh = grad_h_phi @ Phi_i @ g_x    # (1,2)
                    
                    if not (np.all(np.isfinite(Lgh)) and np.all(np.isfinite(Lfh))):
                        continue

                    rhs = float(Lfh + self.alpha * h_tilde)

                    # If control has almost no effect but rhs < 0, skip this
                    if np.linalg.norm(Lgh) < 1e-9 and rhs < 0.0:
                        if self.debug:
                            print(f"[occ] skip degenerate (||Lgh||≈0, rhs={rhs:.3e}<0)")
                        continue
                    
                    # CBF form: L_f h + L_g h u + α h ≥ 0  →  -L_g h u ≤ L_f h + α h
                    A_list.append(-Lgh)
                    b_list.append(np.array([[rhs]]))

                    # if (np.all(np.isfinite(Lgh)) and
                    #     np.all(np.isfinite(Lfh)) and
                    #     np.isfinite(h_tilde)):
                    #     A_list.append(-Lgh)
                    #     b_list.append(Lfh + self.alpha * h_tilde)
                    
        # 6) Backup set constraint at final time
        phi_T = phi_b[-1].reshape(-1, 1)
        Phi_T = Phi_b[-1]
        
        # self.robot.set_terminal_backup_context(
        #     None if no_occ else self.occlusion_scenarios,
        #     self.T_horizon,
        #     kappa=self.kappa,
        #     rho_T=0.1
        # )
        term_scn = None if no_occ else self.occlusion_scenarios[0]
        self.robot.set_terminal_backup_context(
            term_scn,
            self.T_horizon,
            kappa=self.kappa,
            rho_T=0.5
        )

        h_b_T = self.robot.h_b_stop(phi_T)
        grad_h_b_T = self.robot.grad_h_b_stop(phi_T)
        # print(f"h_b_T: {h_b_T}, grad_h_b_T: {grad_h_b_T}")

        Lfh_b_T = grad_h_b_T @ Phi_T @ self.robot.f(robot_state)
        Lgh_b_T = grad_h_b_T @ Phi_T @ self.robot.g(robot_state)

        if np.all(np.isfinite(Lgh_b_T)) and np.all(np.isfinite(Lfh_b_T)) and np.all(np.isfinite(h_b_T)):
            # A_list.append(-Lgh_b_T)
            # b_list.append(Lfh_b_T + self.alpha * h_b_T)
            rhs_b = float(Lfh_b_T + self.alpha * h_b_T)

            # Skip degenerate constraint that would be infeasible by construction
            if np.linalg.norm(Lgh_b_T) < 1e-9 and rhs_b < 0.0:
                if self.debug:
                    print(f"[backup] skip degenerate (||Lgh||≈0, rhs={rhs_b:.3e}<0)")
            else:
                # print("loop_feasible in")
                A_list.append(-Lgh_b_T)
                b_list.append(np.array([[rhs_b]]))

        # 7) If no constraints, use nominal control
        num_constraints = len(A_list)
        # print(f"num_cons: {num_constraints}")
        if num_constraints == 0:
            self.status = 'optimal'
            if self.debug:
                print("[BackupCBFQP] no constraints -> use u_ref")
            return self.u_ref.value

        # Stack constraints into parameter matrices
        A_cbf_val = np.vstack(A_list).reshape(num_constraints, 2)
        b_cbf_val = np.vstack(b_list).reshape(num_constraints, 1)

        self.A_cbf.value[:, :] = 0.0
        self.b_cbf.value[:, :] = 1e6
        self.A_cbf.value[:num_constraints, :] = A_cbf_val
        self.b_cbf.value[:num_constraints, :] = b_cbf_val
        
        if self.debug:
            viol = float(np.max(A_cbf_val @ self.u_ref.value - b_cbf_val))
            print(f"[BackupCBFQP] num_constraints={num_constraints}, max_viol(u_ref)={viol:.3e}")

        # 8) Solve QP (try GUROBI first, fall back to OSQP)
        t_start_qp = time.perf_counter()
        try:
            self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
        except cp.error.SolverError:
            self.cbf_controller.solve(solver=cp.OSQP)
            
        t_end_qp = time.perf_counter()
        
        qp_solve_time_ms = (t_end_qp - t_start_qp) * 1000
        
        if self.debug:
            print(f"[BackupCBFQP] QP Solve Time: {qp_solve_time_ms:.3f} ms")
            
        self.status = self.cbf_controller.status

        if self.status != 'optimal':
            # print("loop1 in")
            # if (not no_occ) and hasattr(self.robot, "backup_input_occlusion"):
            #     print("loop2 in")
            #     return self.robot.backup_input_occlusion(robot_state, self.occlusion_scenarios)
            # elif hasattr(self.robot, "backup_input"):
            #     print("loop3 in")
            #     return self.robot.backup_input(robot_state)
            # elif hasattr(self.robot, "stop"):
            #     print("loop4 in")
            #     return self.robot.stop(robot_state)
            # else:
            #     print("loop5 in")
            #     return np.zeros((2, 1))
            raise InfeasibleError("QP infeasible")
            
        # if self.status != 'optimal':
        #     return (self.robot.backup_input_occlusion(robot_state, self.occlusion_scenarios)
        #         if not no_occ else
        #         self.robot.backup_input(robot_state))
        #     # return self.robot.backup_input(robot_state)

        return self.u.value