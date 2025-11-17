from robots.robot import BaseRobot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely.geometry import Polygon, Point, LineString
from shapely import is_valid_reason
from utils.geometry import custom_merge

class BaseRobotDyn(BaseRobot):

    def __init__(self, X0, robot_spec, dt, ax):
        super().__init__(X0, robot_spec, dt, ax)

        self.collision_parabola_patches = []
        self.collision_parabola_patch = None
        
        self.collision_cone_patches = []
        self.collision_cone_patch = None

        self.rel_vel_patches = []
        
        # Occlusion visualization handles
        self.occlusion_patches = []
        self.occlusion_softmax_contours = []

    def draw_collision_cone(self, X, obs_list, ax):
        '''
        Render the collision cone based on phi
        obs: [obs_x, obs_y, obs_r]
        '''
        if self.robot_spec['model'] != 'KinematicBicycle2D_C3BF':
            return
        
        # Remove previous collision cones safely
        if not hasattr(self, 'collision_cone_patches'):
            self.collision_cone_patches = [] # Initialize attribute
        
        # Remove previous relative vel safely
        if not hasattr(self, 'rel_vel_patches'):
            self.rel_vel_patches = []

        for patch in list(self.collision_cone_patches):
            if patch in ax.patches:
                patch.remove()
        self.collision_cone_patches.clear()

        for arrow in self.rel_vel_patches:
            arrow.remove()
        self.rel_vel_patches.clear()

        # Robot and obstacle positions
        robot_pos = self.get_position()
        theta = X[2, 0]
        v = X[3, 0]

        obstacles_with_dist = []
        for obs in obs_list:
            obs_pos = np.array([obs[0], obs[1]])
            distance = np.linalg.norm(obs_pos - robot_pos)
            obstacles_with_dist.append((distance, obs))
        
        obstacles_with_dist.sort(key=lambda item: item[0])

        num_to_plot = min(20, len(obstacles_with_dist))
        closest_obs_list = [item[1] for item in obstacles_with_dist[:num_to_plot]]
        
        if num_to_plot > 0:
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_to_plot))
        else:
            colors = []

        for i, obs in enumerate(closest_obs_list):
            obs = np.array(obs).flatten()
            obs_pos = np.array([obs[0], obs[1]])
            obs_radius = obs[2]
            obs_vel_x = obs[3]
            obs_vel_y = obs[4]
            beta = 1.05

            # Combine radius R
            ego_dim = obs_radius + self.robot_spec['radius'] * beta # max(c1,c2) + robot_width/2

            v = X[3, 0]
            p_rel = np.array([[obs[0] - X[0, 0]], 
                        [obs[1] - X[1, 0]]])
            v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                        [obs_vel_y - v * np.sin(theta)]])

            p_rel_mag = np.linalg.norm(p_rel)

            # Calculate Collision cone angle
            phi = np.arcsin(ego_dim / p_rel_mag)

            cone_dir = -p_rel / p_rel_mag
            rot_matrix_left = np.array([[np.cos(phi), -np.sin(phi)],
                                        [np.sin(phi),  np.cos(phi)]])
            rot_matrix_right = np.array([[np.cos(-phi), -np.sin(-phi)],
                                        [np.sin(-phi),  np.cos(-phi)]])
            cone_left = (rot_matrix_left @ cone_dir).flatten()
            cone_right = (rot_matrix_right @ cone_dir).flatten()

            # Extend cone boundaries
            cone_left = (robot_pos + 4 * cone_left).tolist()
            cone_right = (robot_pos + 4 * cone_right).tolist()

            # Draw the cone
            cone_points = np.array ([robot_pos.tolist(), cone_left, cone_right])
            collision_cone_patch = patches.Polygon( # only edgecolors different
                cone_points, closed = True,
                edgecolor=colors[i], linestyle='--', alpha=0.5, label=f"Obstacle {i}"
            )
            ax.add_patch(collision_cone_patch)
            self.collision_cone_patches.append(collision_cone_patch)

            offset_angle = 0.003 * (i - (len(obs_list)//2))
            R_offset = np.array([
                [np.cos(offset_angle), -np.sin(offset_angle)],
                [np.sin(offset_angle),  np.cos(offset_angle)]
            ])
            v_rel_offset = R_offset @ v_rel

            arrow = ax.arrow(float(robot_pos[0]), float(robot_pos[1]),
                            float(v_rel_offset[0]), float(v_rel_offset[1]),
                            color=colors[i], width=0.01, alpha=1.0)
            self.rel_vel_patches.append(arrow)

    def draw_collision_parabola(self, X, obs_list, ax):
        '''
        Render the collision parabola based on functions
        obs: [obs_x, obs_y, obs_r]
        '''
        if self.robot_spec['model'] not in ['KinematicBicycle2D_DPCBF']:
            return

        # Remove previous collision parabolas safely
        if not hasattr(self, 'collision_parabola_patches'):
            self.collision_parabola_patches = [] # Initialize attribute

        # Remove previous relative vel safely
        if not hasattr(self, 'rel_vel_patches'):
            self.rel_vel_patches = []

        for line in self.collision_parabola_patches:
                line.remove()
        self.collision_parabola_patches.clear()

        for arrow in self.rel_vel_patches:
                arrow.remove()
        self.rel_vel_patches.clear()

        # Robot and obstacle positions
        robot_pos = self.get_position()

        obstacles_with_dist = []
        for obs in obs_list:
            obs_pos = np.array([obs[0], obs[1]])
            distance = np.linalg.norm(obs_pos - robot_pos)
            obstacles_with_dist.append((distance, obs))

        obstacles_with_dist.sort(key=lambda item: item[0])

        num_to_plot = min(20, len(obstacles_with_dist))
        closest_obs_list = (item[1] for item in obstacles_with_dist[:num_to_plot])
        
        if num_to_plot > 0:
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_to_plot))
        else:
            colors = []

        for i, obs in enumerate(closest_obs_list):

            theta = X[2, 0]
            v = X[3, 0]

            obs = np.array(obs).flatten()
            obs_pos = np.array([obs[0], obs[1]])
            obs_radius = obs[2]
            obs_vel_x = obs[3]
            obs_vel_y = obs[4]

            # safety margin
            beta = 1.05
            # Combine radius R
            ego_dim = (obs_radius + self.robot_spec['radius']) * beta # max(c1,c2) + robot_width/2 (we suppose safe r as radius)

            p_rel = obs_pos - robot_pos
            v_rel = np.array([[obs_vel_x - v * np.cos(theta)], 
                            [obs_vel_y - v * np.sin(theta)]])

            p_rel_mag = np.linalg.norm(p_rel)
            v_rel_mag = np.linalg.norm(v_rel)

            # Compute d_safe safely
            eps = 1e-6
            d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)

            # DPCBF functions
            k_lambda, k_mu = 0.1 * np.sqrt(beta**2 - 1)/ego_dim, 0.5 * np.sqrt(beta**2 - 1)/ego_dim
            func_lambda = k_lambda * np.sqrt(d_safe) / v_rel_mag
            func_mu = k_mu * np.sqrt(d_safe)

            rot_angle = np.arctan2(p_rel[1], p_rel[0])
            R = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                        [-np.sin(rot_angle),  np.cos(rot_angle)]])

            L = 1.5
            y_disp = np.linspace(-L, L, 100)
            x_disp = (-func_lambda * (y_disp**2) - func_mu)

            pts_world = robot_pos.reshape(2,1) + R.T @ np.vstack([x_disp, y_disp])
            line, = ax.plot(pts_world[0,:], pts_world[1, :],
                            color=colors[i], linestyle='-', linewidth=2.0,
                            label=f"Quadratic Obs {i}")
            self.collision_parabola_patches.append(line)

            offset_angle = 0.02 * (i - (len(obs_list)//2))
            R_offset = np.array([
                [np.cos(offset_angle), -np.sin(offset_angle)],
                [np.sin(offset_angle),  np.cos(offset_angle)]
            ])
            v_rel_offset = R_offset @ v_rel

            x_offset = 1.0
            y_offset = 1.0
        
            arrow = ax.arrow(float(robot_pos[0]), float(robot_pos[1]),
                            float(x_offset * v_rel_offset[0]), float(y_offset * v_rel_offset[1]),
                            color=colors[i], width=0.02, alpha=1.0)
            self.rel_vel_patches.append(arrow)
            
    def _build_occlusion_U0(self, p, c, R_o, R_s, t1, t2, n_arc=40):
        """
        Construct the exact initial occlusion set U0 as:
          - outer arc : circle centered at p, radius R_s
          - inner arc : circle centered at c, radius R_o
          - 2 tangential lines connecting arcs via t1, t2

        Returns:
            poly : (N,2) polygon vertices in CCW order, or None
        """
        import numpy as np
        import math

        p = np.asarray(p, float)
        c = np.asarray(c, float)
        t1 = np.asarray(t1, float)
        t2 = np.asarray(t2, float)

        def norm(v):
            return float(np.linalg.norm(v))

        def normalize_angle(a):
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        def arc_with_mid(a1, a2, amid, n):
            # choose orientation so that 'amid' is on the arc
            a1 = normalize_angle(a1)
            a2 = normalize_angle(a2)
            amid = normalize_angle(amid)

            # ccw span
            ccw = normalize_angle(a2 - a1)
            if ccw <= 0:
                ccw += 2.0 * math.pi

            d_mid = normalize_angle(amid - a1)
            if 0.0 <= d_mid <= ccw:
                # use ccw
                return np.linspace(a1, a1 + ccw, n)
            else:
                # use cw (complement)
                cw = 2.0 * math.pi - ccw
                return np.linspace(a1, a1 - cw, n)

        # directions from p through tangent points -> outer intersection
        v1 = t1 - p
        v2 = t2 - p
        if norm(v1) < 1e-9 or norm(v2) < 1e-9:
            return None
        d1 = v1 / norm(v1)
        d2 = v2 / norm(v2)

        f1 = p + R_s * d1
        f2 = p + R_s * d2

        # outer arc angles (center p)
        phi1 = math.atan2(f1[1] - p[1], f1[0] - p[0])
        phi2 = math.atan2(f2[1] - p[1], f2[0] - p[0])
        phi_c = math.atan2(c[1] - p[1], c[0] - p[0])  # direction to obstacle

        outer_phis = arc_with_mid(phi1, phi2, phi_c, n_arc)
        outer_arc = np.vstack([
            p[0] + R_s * np.cos(outer_phis),
            p[1] + R_s * np.sin(outer_phis)
        ]).T

        # inner arc on obstacle circle: choose orientation that does NOT include robot
        theta1 = math.atan2(t1[1] - c[1], t1[0] - c[0])
        theta2 = math.atan2(t2[1] - c[1], t2[0] - c[0])
        theta_p = math.atan2(p[1] - c[1], p[0] - c[0])

        arc_inc = arc_with_mid(theta1, theta2, theta_p, n_arc)
        # complementary arc excludes robot direction
        arc_exc = arc_inc[::-1]

        inner_arc = np.vstack([
            c[0] + R_o * np.cos(arc_exc),
            c[1] + R_o * np.sin(arc_exc)
        ]).T

        # Orient inner_arc so that it starts near t2 (for a nice closed loop)
        if np.linalg.norm(inner_arc[0] - t2) > np.linalg.norm(inner_arc[-1] - t2):
            inner_arc = inner_arc[::-1]

        # Build polygon: outer arc(f1→f2) -> line to t2 -> inner arc(t2→t1) -> line to f1
        poly = []
        poly.extend(outer_arc.tolist())
        poly.append(t2.tolist())
        poly.extend(inner_arc.tolist())
        poly.append(t1.tolist())

        return np.asarray(poly)
    
    def _curved_softmax_value(self, pos, scenario, R_s, kappa):
        """
        h̃(x) from the same definition as BackupCBFQP._occlusion_barrier_softmax_curved
        but at tau = 0, using:
          - first two halfspaces (tangent lines)
          - sensing circle
          - obstacle circle
        """
        import numpy as np

        A = scenario.get('A', None)
        b0 = scenario.get('b0', None)
        if A is None or b0 is None or A.shape[0] < 2:
            return None

        p = scenario['robot_pos']
        c = scenario['obs_center']
        R_o = scenario['obs_radius']
        R = self.robot_radius

        pos = np.asarray(pos, float)
        x, y = pos

        a1, a2 = A[3], A[1]
        beta1, beta2 = b0[3], b0[1]

        h1 = a1 @ pos - beta1 - R
        h2 = a2 @ pos - beta2 - R

        dx_s, dy_s = x - p[0], y - p[1]
        h3 = dx_s*dx_s + dy_s*dy_s - (R_s + R)**2

        dx_o, dy_o = x - c[0], y - c[1]
        h4 = dx_o*dx_o + dy_o*dy_o - (R_o + R)**2
        
        h_vec = np.array([h1, h2, h3, h4], dtype=float)
        if not np.all(np.isfinite(h_vec)):
            return None

        M = h_vec.size
        if not np.all(np.isfinite(h_vec)):
            return None

        max_h = np.max(h_vec)
        z = np.exp(kappa * (h_vec - max_h))
        Z = np.sum(z)
        if not np.isfinite(Z) or Z <= 0.0:
            return None

        lse = max_h + np.log(Z)
        return float((lse - np.log(M)) / kappa)

    
    def update_occlusion_polygons(self, occlusion_scenarios,
                                  use_curved=True,
                                  kappa=10.0,
                                  show_true_occ=True,
                                  grid_res=0.05):

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if use_curved==False and show_true_occ==False:
            return

        # 1) reset previous patches
        for p in self.occlusion_patches:
            try:
                p.remove()
            except:
                pass
        self.occlusion_patches = []

        for coll in self.occlusion_softmax_contours:
            try:
                coll.remove()
            except:
                pass
        self.occlusion_softmax_contours = []

        if not occlusion_scenarios:
            return

        for sc in occlusion_scenarios:
            poly = sc.get('poly', None)
            t1 = sc.get('t1', None)
            t2 = sc.get('t2', None)
            if poly is None or t1 is None or t2 is None:
                continue

            p = np.asarray(sc['robot_pos'], float)
            c = np.asarray(sc['obs_center'], float)
            R_o = float(sc['obs_radius'])

            if poly.shape[0] >= 4:
                far = np.asarray(poly[3], float)
            else:
                far = np.asarray(poly[2], float)
            R_s = float(np.linalg.norm(far - p))

            # 1) geometric real occluded region
            U0 = self._build_occlusion_U0(p, c, R_o, R_s, t1, t2)
            if show_true_occ and U0 is not None:
                patch = patches.Polygon(
                    U0,
                    closed=True,
                    fill=True,
                    facecolor='gray',
                    edgecolor='none',
                    alpha=0.25,
                    zorder=1
                )
                self.ax.add_patch(patch)
                self.occlusion_patches.append(patch)

            # 2) softmax over-approximate boundary
            if use_curved:
                if U0 is not None:
                    xmin, xmax = U0[:, 0].min(), U0[:, 0].max()
                    ymin, ymax = U0[:, 1].min(), U0[:, 1].max()
                else:
                    xmin, xmax = poly[:, 0].min(), poly[:, 0].max()
                    ymin, ymax = poly[:, 1].min(), poly[:, 1].max()

                pad = 1
                xs = np.arange(xmin - pad, xmax + pad, grid_res)
                ys = np.arange(ymin - pad, ymax + pad, grid_res)
                XX, YY = np.meshgrid(xs, ys)

                h_field = np.full_like(XX, np.nan, dtype=float)
                for i in range(XX.shape[0]):
                    for j in range(XX.shape[1]):
                        val = self._curved_softmax_value(
                            (XX[i, j], YY[i, j]),
                            sc, R_s, kappa
                        )
                        if val is not None and np.isfinite(val):
                            h_field[i, j] = val

                try:
                    import numpy as np

                    M = 3.0  # num convex constraints
                    level = -np.log(M) / kappa # optional

                    cs = self.ax.contour(
                        XX, YY, h_field,
                        levels=[0.0],
                        colors='red',
                        linewidths=2.0,
                        zorder=2
                    )
                    # cs = self.ax.contour(
                    #     XX, YY, h_field,
                    #     levels=[0.0],
                    #     colors='red',
                    #     linewidths=2.0,
                    #     zorder=2
                    # )
                    
                    for coll in cs.collections:
                        self.occlusion_softmax_contours.append(coll)
                except Exception:
                    pass

    def set_occ_barrier_fn(self, fn):
        """
        Forward occlusion barrier callback into the underlying robot model
        (e.g., DoubleIntegrator2D).
        """
        if hasattr(self.robot, "set_occ_barrier_fn"):
            self.robot.set_occ_barrier_fn(fn)
        else:
            raise AttributeError("Underlying robot does not support set_occ_barrier_fn")
        
    # robots/base_robot_dyn.py (혹은 BaseRobotDyn 정의된 파일)

    def set_terminal_backup_context(self, occlusion_scenarios, T, kappa=None, rho_T=0.05):
        """
        BackupCBFQP가 터미널 백업 컨텍스트를 주입할 때
        내부 실제 로봇(예: DoubleIntegrator2D)으로 전달.
        내부 모델에 해당 메서드가 없으면 안전한 no-op 컨텍스트를 저장해 둠.
        """
        inner = getattr(self, "robot", None)
        if inner is not None and hasattr(inner, "set_terminal_backup_context"):
            return inner.set_terminal_backup_context(occlusion_scenarios, T, kappa=kappa, rho_T=rho_T)

        # fallback no-op (velocity 기반 백업으로 자동 폴백되도록 상태만 저장)
        self._term_occ_scenarios = occlusion_scenarios if occlusion_scenarios else []
        self._term_T = float(T)
        self._term_kappa = kappa
        self._term_rho = float(rho_T)
        return None

    def h_b_stop(self, X):
        """
        터미널 백업 집합 값 h_b(x_T).
        내부 모델이 구현했으면 그대로 위임, 없으면 속도 기반 폴백을 사용.
        """
        inner = getattr(self, "robot", None)
        if inner is not None and hasattr(inner, "h_b_stop"):
            return inner.h_b_stop(X)

        # fallback: v_safe^2 - ||v||^2
        v_sq = float(X[2, 0]**2 + X[3, 0]**2)
        v_safe_sq = (0.3)**2
        return v_safe_sq - v_sq

    def grad_h_b_stop(self, X):
        """
        h_b_stop의 그래디언트.
        내부 모델이 구현했으면 그대로 위임, 없으면 속도 기반 폴백의 기울기 사용.
        """
        inner = getattr(self, "robot", None)
        if inner is not None and hasattr(inner, "grad_h_b_stop"):
            return inner.grad_h_b_stop(X)

        return np.array([[0, 0, -2 * X[2, 0], -2 * X[3, 0]]])
