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
        self.occlusion_future_contours = []
        self._occ_arc_lines = []
        self._occ_barrier_cb = None

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
            
    def _circle_tangents_local(self, p, c, R):
        p = np.asarray(p, float).reshape(2,)
        c = np.asarray(c, float).reshape(2,)
        v = p - c
        d2 = float(v @ v)
        R2 = R * R
        if d2 <= R2:   # 로봇이 확장원 안/위면 접선 없음
            return None, None
        x1, y1 = v
        x0 = R2 * x1 / d2
        y0 = R2 * y1 / d2
        k = R * np.sqrt(d2 - R2) / d2
        t1 = np.array([x0 - y1 * k, y0 + x1 * k]) + c
        t2 = np.array([x0 + y1 * k, y0 - x1 * k]) + c
        return t1, t2

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
    
    def _line_circle_intersections(self, n, beta, p, R):
        """
        n: (2,) 단위 법선벡터
        선: n·x = beta
        원: 중심 p, 반지름 R
        return: 교점 2개 (x_plus, x_minus) 또는 None
        """
        import numpy as np
        n = np.asarray(n, float).reshape(2,)
        p = np.asarray(p, float).reshape(2,)
        d = float(beta - n @ p)               # 원 중심에서 선까지 부호거리
        if abs(d) > R + 1e-9:
            return None
        # 선 위의 중심에 가장 가까운 점
        x0 = p + d * n
        # 선의 방향단위벡터 (법선에 수직)
        t = np.array([-n[1], n[0]], dtype=float)
        l = float(np.sqrt(max(R*R - d*d, 0.0)))
        return x0 + l * t, x0 - l * t

    def _build_occlusion_UT_offset(self, p, c, R_o, R_s, t1, t2, d, n_arc=60):
        """
        U0의 두 접선( p->t1, p->t2 )을 각 선의 '장애물쪽' 법선으로 거리 d 만큼
        평행이동하여 U_T 경계를 구성.
        내부 원호는 R_o+d, 외부 원호는 R_s 유지.
        """
        import numpy as np, math

        p = np.asarray(p, float); c = np.asarray(c, float)
        t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)

        def arc_with_mid(a1, a2, amid, n):
            def normang(a): return (a + math.pi) % (2*math.pi) - math.pi
            a1 = normang(a1); a2 = normang(a2); amid = normang(amid)
            ccw = normang(a2 - a1);  ccw = ccw + 2*math.pi if ccw <= 0 else ccw
            dmid = normang(amid - a1)
            # amid(=장애물 방향)을 포함하는 호를 선택
            if 0.0 <= dmid <= ccw:
                return np.linspace(a1, a1 + ccw, n)
            else:
                cw = 2*math.pi - ccw
                return np.linspace(a1, a1 - cw, n)

        # (1) U0 접선의 법선 = (t_i - c)/||t_i - c||  (접점의 반지름 방향 = 접선의 법선)
        n1 = t1 - c; n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = t2 - c; n2 = n2 / (np.linalg.norm(n2) + 1e-12)

        # U0 접선식: n_i · x = beta0_i,  beta0_i = n_i · t_i
        beta01 = float(n1 @ t1)
        beta02 = float(n2 @ t2)

        # (2) 법선방향으로 d 만큼 평행이동 → beta' = beta0 + d
        beta1 = beta01 + d
        beta2 = beta02 + d
        R_eff = R_o + d

        # (3) 내부(확장) 원과의 접점: q_i' = c + R_eff * n_i
        q1 = c + R_eff * n1
        q2 = c + R_eff * n2

        # (4) 외부 원과의 교점: n_i·x = beta_i 와 O(p, R_s) 교점 중,
        #     p→t_i 방향에 더 가까운 것을 선택
        u1 = (t1 - p); u1 = u1 / (np.linalg.norm(u1) + 1e-12)
        u2 = (t2 - p); u2 = u2 / (np.linalg.norm(u2) + 1e-12)

        ints1 = self._line_circle_intersections(n1, beta1, p, R_s)
        ints2 = self._line_circle_intersections(n2, beta2, p, R_s)
        if (ints1 is None) or (ints2 is None):
            return None  # 센싱반경 밖이면 그리지 않음

        cand11, cand12 = ints1
        cand21, cand22 = ints2
        # 방향 선택: p에서 교점으로의 단위벡터가 u_i와 내적 최대인 점
        def pick_dir(c1, c2, u):
            v1 = c1 - p; v1 = v1 / (np.linalg.norm(v1) + 1e-12)
            v2 = c2 - p; v2 = v2 / (np.linalg.norm(v2) + 1e-12)
            return c1 if (v1 @ u) >= (v2 @ u) else c2

        f1 = pick_dir(cand11, cand12, u1)  # 외부 원의 끝점 (L1')
        f2 = pick_dir(cand21, cand22, u2)  # 외부 원의 끝점 (L2')

        # (5) 외부 원호 (f1→f2), 장애물 방향을 포함하는 쪽으로
        phi1 = math.atan2(f1[1]-p[1], f1[0]-p[0])
        phi2 = math.atan2(f2[1]-p[1], f2[0]-p[0])
        phi_c = math.atan2(c[1]-p[1],  c[0]-p[0])
        outer_phis = arc_with_mid(phi1, phi2, phi_c, n_arc)
        outer_arc = np.vstack([p[0] + R_s*np.cos(outer_phis),
                            p[1] + R_s*np.sin(outer_phis)]).T

        # (6) 내부(확장) 원호: 로봇 방향 포함하지 않는 보완호 (q2→q1)
        th1 = math.atan2(q1[1]-c[1], q1[0]-c[0])
        th2 = math.atan2(q2[1]-c[1], q2[0]-c[0])
        thp = math.atan2(p[1]-c[1],  p[0]-c[0])
        inc = arc_with_mid(th1, th2, thp, n_arc)   # 로봇방향 포함하는 호
        exc = inc[::-1]                             # 보완호 (로봇방향 제외)
        inner_arc = np.vstack([c[0] + R_eff*np.cos(exc),
                            c[1] + R_eff*np.sin(exc)]).T
        # inner_arc 시작을 q2에 맞춰 방향 q2→q1로 정렬
        if np.linalg.norm(inner_arc[0]-q2) > np.linalg.norm(inner_arc[-1]-q2):
            inner_arc = inner_arc[::-1]

        # (7) 최종 다각형: outer_arc(f1→f2) → q2 → inner_arc(q2→q1) → q1
        poly = []
        poly.extend(outer_arc.tolist())
        poly.append(q2.tolist())
        poly.extend(inner_arc.tolist())
        poly.append(q1.tolist())
        return np.asarray(poly), (outer_arc, inner_arc, f1, f2, q1, q2)
    
    def _ut_halfspaces_params(self, p, c, R_o, t1, t2, d):
        """
        U0의 접점(t1,t2)을 기준으로, τ=T에서 평행이동된 두 접선의
        (법선 n_i, 오프셋 beta_i)와 확장반경 R_eff를 반환.
        """
        import numpy as np
        p = np.asarray(p, float); c = np.asarray(c, float)
        t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)

        n1 = t1 - c; n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = t2 - c; n2 = n2 / (np.linalg.norm(n2) + 1e-12)

        beta01 = float(n1 @ t1)
        beta02 = float(n2 @ t2)
        beta1 = beta01 + float(d)
        beta2 = beta02 + float(d)

        R_eff = float(R_o + d)
        return (n1, beta1), (n2, beta2), R_eff
    
    def _softmin_field_UT(self, XX, YY, p, c, R_s, n1, beta1, n2, beta2, R_eff, kappa):
        R = float(self.robot_radius)

        # c_i(x) >= 0 inside UT
        c1 = (beta1 + R) - (XX * n1[0] + YY * n1[1])                 # halfspace 1
        c2 = (beta2 + R) - (XX * n2[0] + YY * n2[1])                 # halfspace 2
        c3 = (R_s + R) - np.hypot(XX - p[0], YY - p[1])              # inside sensing disc
        c4 = np.hypot(XX - c[0], YY - c[1]) - (R_eff + R)            # outside expanded obstacle

        C = np.stack([c1, c2, c3, c4], axis=2)   # (..., 4)

        # softmin_kappa(C) = Cmin - (1/kappa) * ( log(sum(exp(-kappa*(C - Cmin)))) - log(4) )
        Cmin = np.min(C, axis=2, keepdims=True)
        sumexp = np.exp(-kappa * (C - Cmin)).sum(axis=2, keepdims=False)
        h_tilde = (Cmin.squeeze(-1)
                - (np.log(sumexp) - np.log(4.0)) / float(kappa))
        return h_tilde
    
    def _softmin_field_UT_masked(self, XX, YY, p, c, R_s,
                                n1, beta1, n2, beta2, R_eff,
                                kappa, wedge_eps=1e-3, band_eps=None):
        """
        h̃_T를 계산하되, UT 경계 근처(밴드) ∩ 웨지 안(c1,c2) ∩
        센싱원 안(c3) ∩ 확장장애물 밖(c4) 만 남기고 나머지는 NaN으로 마스킹.
        """
        R = float(self.robot_radius)

        # 각 제약의 'inside' SDF (>=0가 내부)
        c1 = (beta1 + R) - (XX * n1[0] + YY * n1[1])              # halfspace 1 (inside when >=0)
        c2 = (beta2 + R) - (XX * n2[0] + YY * n2[1])              # halfspace 2
        c3 = (R_s + R) - np.hypot(XX - p[0], YY - p[1])           # inside sensing disc
        c4 = np.hypot(XX - c[0], YY - c[1]) - (R_eff + R)         # outside expanded obstacle

        # softmin (교집합용) – 기존과 동일
        C = np.stack([c1, c2, c3, c4], axis=2)                    # (..., 4)
        Cmin = np.min(C, axis=2, keepdims=True)
        h_tilde = ( Cmin.squeeze(-1)
                    - ( np.log(np.exp(-kappa*(C - Cmin)).sum(axis=2)) - np.log(4.0) ) / float(kappa) )

        # ----- 마스크 -----
        # 웨지 내부(두 반평면 만족). 약간의 여유를 둠.
        mask_wedge = (c1 >= -wedge_eps) & (c2 >= -wedge_eps)
        # 원 조건도 너무 빡세지 않게 작은 여유
        ring_eps = 3.0 * max(1e-3, float((XX[0,1]-XX[0,0])))      # ≈ 3 * grid_res
        mask_rings = (c3 >= -ring_eps) & (c4 >= -ring_eps)

        # 경계 근처만 띠로 남김(softmin=0 부근)
        if band_eps is None:
            band_eps = 2.5 * max(1e-3, float((XX[0,1]-XX[0,0])))  # ≈ 2.5 * grid_res
        mask_band = np.abs(h_tilde) <= band_eps

        mask = mask_wedge & mask_rings & mask_band

        Z = h_tilde.copy()
        Z[~mask] = np.nan
        return Z

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

        # dx_s, dy_s = x - p[0], y - p[1]
        # h3 = dx_s*dx_s + dy_s*dy_s - (R_s + R)**2

        # dx_o, dy_o = x - c[0], y - c[1]
        # h4 = dx_o*dx_o + dy_o*dy_o - (R_o + R)**2
        dx_s, dy_s = x - p[0], y - p[1]
        d_s = np.hypot(dx_s, dy_s)
        h3 = d_s - (R_s + R)           # 센싱 원호

        dx_o, dy_o = x - c[0], y - c[1]
        d_o = np.hypot(dx_o, dy_o)
        h4 = d_o - (R_o + R)           # 장애물 원호
        
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

    def _clear_artists(self, lst):
        for a in lst:
            try:
                # contour/contourf 인 경우
                if hasattr(a, "collections"):
                    for c in a.collections:
                        try: c.remove()
                        except: pass
                else:
                    a.remove()
            except Exception:
                pass
        lst.clear()
        
    def update_occlusion_polygons(self, occlusion_scenarios,
                                  use_curved=False,
                                  kappa=10.0,
                                  show_true_occ=True,
                                  show_true_occ_T=True,
                                  show_softmax_occ_T=True,
                                  T_rollout=None,
                                  grid_res=0.05):
        
        # 1) reset previous patches
        self._clear_artists(self.occlusion_patches)
        self._clear_artists(self.occlusion_softmax_contours)
        self._clear_artists(self.occlusion_future_contours)
        self._clear_artists(self._occ_arc_lines)
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
        
        for coll in getattr(self, "occlusion_future_contours", []):  # <-- U_T 초기화
            try: coll.remove()
            except: pass
        self.occlusion_future_contours = []

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

            # # 2) softmax over-approximate boundary
            # if use_curved:
            #     if U0 is not None:
            #         xmin, xmax = U0[:, 0].min(), U0[:, 0].max()
            #         ymin, ymax = U0[:, 1].min(), U0[:, 1].max()
            #     else:
            #         xmin, xmax = poly[:, 0].min(), poly[:, 0].max()
            #         ymin, ymax = poly[:, 1].min(), poly[:, 1].max()

            #     pad = 1
            #     xs = np.arange(xmin - pad, xmax + pad, grid_res)
            #     ys = np.arange(ymin - pad, ymax + pad, grid_res)
            #     XX, YY = np.meshgrid(xs, ys)

            #     h_field = np.full_like(XX, np.nan, dtype=float)
            #     for i in range(XX.shape[0]):
            #         for j in range(XX.shape[1]):
            #             val = self._curved_softmax_value(
            #                 (XX[i, j], YY[i, j]),
            #                 sc, R_s, kappa
            #             )
            #             if val is not None and np.isfinite(val):
            #                 h_field[i, j] = val

            #     try:
            #         M = 3.0  # num convex constraints
            #         level = -np.log(M) / kappa # optional

            #         cs = self.ax.contour(
            #             XX, YY, h_field,
            #             levels=[0.0],
            #             colors='red',
            #             linewidths=2.0,
            #             zorder=2
            #         )
            #         # cs = self.ax.contour(
            #         #     XX, YY, h_field,
            #         #     levels=[0.0],
            #         #     colors='red',
            #         #     linewidths=2.0,
            #         #     zorder=2
            #         # )
                    
            #         for coll in cs.collections:
            #             self.occlusion_softmax_contours.append(coll)
            #     except Exception:
            #         pass
                
            # ---- 3) U_T (future occlusion) : U0 접선의 법선방향 offset 구현 ----
            if show_true_occ_T and (T_rollout is not None) and (T_rollout > 0.0):
                v_adv = float(sc.get('v_adv_max', 0.5))
                d = float(v_adv * T_rollout)

                built = self._build_occlusion_UT_offset(p, c, R_o, R_s, t1, t2, d, n_arc=80)
                if built is not None:
                    UT, aux = built
                    patchT = patches.Polygon(
                        UT, closed=True, fill=True,
                        facecolor="#e41d45", edgecolor='none',
                        alpha=0.22, zorder=0.9
                    )
                    self.ax.add_patch(patchT)
                    self.occlusion_future_contours.append(patchT)

                    # # (선택) 곡률 강조용 점선 호만 얇게 덧그리기
                    # outer_arc, inner_arc, f1, f2, q1, q2 = aux
                    # line1, = self.ax.plot(outer_arc[:,0], outer_arc[:,1],
                    #                     linestyle=':', linewidth=1.6, color='#2b6cb0', zorder=3)
                    # line2, = self.ax.plot(inner_arc[:,0], inner_arc[:,1],
                    #                     linestyle=':', linewidth=1.6, color='#2b6cb0', zorder=3)
                    # self._occ_arc_lines.extend([line1, line2])
                # 접선-센싱 원 교점이 없으면(드묾) 아무 것도 그리지 않음 (전체 원 금지)
            if show_softmax_occ_T:
                for sc_idx, sc in enumerate(occlusion_scenarios):

                    # --- 여기부터 추가: τ=T softmin 경계 그리기 ---
                    (n1b, beta1), (n2b, beta2), R_eff = self._ut_halfspaces_params(p, c, R_o, t1, t2, d)

                    xmin, xmax = UT[:, 0].min(), UT[:, 0].max()
                    ymin, ymax = UT[:, 1].min(), UT[:, 1].max()
                    pad = 0.75
                    xs = np.arange(xmin - pad, xmax + pad, grid_res)
                    ys = np.arange(ymin - pad, ymax + pad, grid_res)
                    XX, YY = np.meshgrid(xs, ys)

                    htilde_T = self._softmin_field_UT(
                        XX, YY, p, c, R_s,
                        n1b, beta1, n2b, beta2, R_eff,
                        kappa=float(kappa)
                    )

                    ut_color = "#f80101ea"

                    csT = self.ax.contour(
                        XX, YY, htilde_T,
                        levels=[0.0],
                        colors=ut_color,
                        linestyles='-',
                        linewidths=1.5,
                        zorder=5
                    )
                    for coll in csT.collections:
                        self.occlusion_softmax_contours.append(coll)
                    
        # built = self._build_occlusion_UT_offset(p, c, R_o, R_s, t1, t2, d, n_arc=80)
        # if built is not None:
        #     UT, aux = built
        #     patchT = patches.Polygon(
        #         UT, closed=True, fill=True,
        #         facecolor="#ffcfd9", edgecolor='none',
        #         linewidth=0.0, alpha=0.22, zorder=0.9
        #     )
        #     patchT.set_antialiased(False)
        #     self.ax.add_patch(patchT)
        #     self.occlusion_future_contours.append(patchT)

        #     # (b) τ=T soft-max 경계 (h̃_T(x)=0) 그리기
        #     #    - UT 범위를 기준으로 그리드 구성
        #     (n1b, beta1), (n2b, beta2), R_eff = self._ut_halfspaces_params(p, c, R_o, t1, t2, d)
        #     xmin, xmax = UT[:, 0].min(), UT[:, 0].max()
        #     ymin, ymax = UT[:, 1].min(), UT[:, 1].max()
        #     pad = 0.75
        #     xs = np.arange(xmin - pad, xmax + pad, grid_res)
        #     ys = np.arange(ymin - pad, ymax + pad, grid_res)
        #     XX, YY = np.meshgrid(xs, ys)

        #     htilde_T = self._softmin_field_UT(
        #         XX, YY, p, c, R_s,
        #         n1b, beta1, n2b, beta2, R_eff,
        #         kappa=float(kappa)
        #     )

        #     csT = self.ax.contour(
        #         XX, YY, htilde_T,
        #         levels=[0.0],
        #         colors="#f80101ea",
        #         linestyles='-',
        #         linewidths=1.5,
        #         zorder=5
        #     )
        #     for coll in csT.collections:
        #         self.occlusion_softmax_contours.append(coll)

    def set_occ_barrier_fn(self, fn):
        """
        Forward occlusion barrier callback into the underlying robot model
        (e.g., DoubleIntegrator2D).
        """
        self._occ_barrier_cb = fn
        inner = getattr(self, "robot", None)
        if inner is not None and hasattr(inner, "set_occ_barrier_fn"):
            inner.set_occ_barrier_fn(fn)
        else:
            raise AttributeError("Underlying robot does not support set_occ_barrier_fn")
        
    def _draw_softmax_levelset(self, scenario, tau, bbox, grid_res, color, lw=1.8, ls='-'):
        """
        scenario: backup_cbf_qp가 쓰는 그 dict (A, b0, robot_pos, obs_center, obs_radius, v_adv_max ...)
        tau     : 0.0 또는 T
        bbox    : (xmin,xmax,ymin,ymax)
        """
        if self._occ_barrier_cb is None:
            return  # 콜백 없으면 스킵

        xmin, xmax, ymin, ymax = bbox
        xs = np.arange(xmin, xmax, grid_res)
        ys = np.arange(ymin, ymax, grid_res)
        XX, YY = np.meshgrid(xs, ys)

        H = np.full_like(XX, np.nan, dtype=float)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                h_val, _, _ = self._occ_barrier_cb((XX[i, j], YY[i, j]), scenario, tau=tau)
                if (h_val is not None) and np.isfinite(h_val):
                    H[i, j] = h_val

        cs = self.ax.contour(
            XX, YY, H,
            levels=[0.0], colors=color, linewidths=lw, linestyles=ls, zorder=5
        )
        for coll in cs.collections:
            self.occlusion_softmax_contours.append(coll)

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
