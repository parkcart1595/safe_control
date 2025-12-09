from tracking import LocalTrackingController
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import subprocess
import csv

class InfeasibleError(Exception):
    '''
    Exception raised for errors when QP is infeasible or 
    the robot collides with the obstacle
    '''

    def __init__(self, message="ERROR in QP or Collision"):
        self.message = message
        super().__init__(self.message)

class LocalTrackingControllerDyn(LocalTrackingController):

    def __init__(self, X0, robot_spec,
                 controller_type=None,
                 dt=0.05,
                 show_animation=False, save_animation=False, show_mpc_traj=False,
                 enable_rotation=True, raise_error=False,
                 ax=None, fig=None, env=None, rand_seed=42):
        super().__init__(X0, robot_spec,
                         controller_type=controller_type,
                         dt=dt,
                         show_animation=show_animation, save_animation=save_animation, show_mpc_traj=show_mpc_traj,
                         enable_rotation=enable_rotation, raise_error=raise_error,
                         ax=ax, fig=fig, env=env)
        
        if self.pos_controller_type == 'cbf_qp':
            from position_control.cbf_qp import CBFQP
            self.pos_controller = CBFQP(self.robot, self.robot_spec, num_obs=10)
        if self.pos_controller_type == 'backup_cbf_qp':
            from position_control.backup_cbf_qp import BackupCBFQP
            self.pos_controller = BackupCBFQP(self.robot, self.robot_spec, num_obs=10)
        
        # Create a list to hold the arrow patches for obstacle velocities
        self.obs_vel_arrows = []
        
        # Initialize moving obstacles
        self.dyn_obs_patch = None # will be initialized after the first step
        self.init_obs_info = None
        self.init_obs_circle = None
        
        self._rng = np.random.default_rng(rand_seed)
        self.obs_meta = None
    
    def setup_robot(self, X0):
        from dynamic_env.robot import BaseRobotDyn
        self.robot = BaseRobotDyn(
            X0.reshape(-1, 1), self.robot_spec, self.dt, self.ax)
        
    # Update dynamic obs position
    # def step_dyn_obs(self):
    #     """if self.obs (n,5) array (ex) [x, y, r, vx, vy], update obs position per time step"""
    #     if len(self.obs) != 0 and self.obs.shape[1] >= 7:
    #         for i in enumerate(self.obs):
    #             x, y, r, vx, vy, y_min, y_max = self.obs[i, :7]
    #             mode = int(self.obs[i, 7]) if self.obs.shape[1] >= 8 else 0
    #             v_max = float(self.obs[i, 8]) if self.obs.shape[1] >= 9 else np.hypot(vx, vy)
    #             theta = float(self.obs[i, 9]) if self.obs.shape[1] >= 10 else float(np.arctan2(vy, vx))

    #             if mode == 1:
    #                 # ---- 랜덤워커: 방향만 부드럽게 요동 ----
    #                 # 작은 방향 노이즈 + 가끔 큰 턴
    #                 dtheta = self._rng.normal(0.0, 0.15)
    #                 if self._rng.random() < 0.05:   # 5% 확률로 큰 방향 전환
    #                     dtheta += self._rng.normal(0.0, 0.7)

    #                 theta += dtheta

    #                 # 속도는 v_max로 유지 (원하면 아래 한 줄을 바꿔 변동 속도도 가능)
    #                 speed = v_max
    #                 vx, vy = speed * np.cos(theta), speed * np.sin(theta)

    #                 # 값 되돌려쓰기
    #                 self.obs[i, 3] = vx
    #                 self.obs[i, 4] = vy
    #                 if self.obs.shape[1] >= 10:
    #                     self.obs[i, 9] = theta
                        
    #             # obs_info = [x, y, r, vx, vy, y_min, y_max]
    #             self.obs[i, 0] += self.obs[i, 3] * self.dt  # x += vx*dt (if vx != 0)
    #             self.obs[i, 1] += self.obs[i, 4] * self.dt  # y += vy*dt

    #             # Flip velocity if it hits top or bottom
    #             y_min = self.obs[i, 5]
    #             y_max = self.obs[i, 6]
    #             if self.obs[i, 1] >= y_max:
    #                 self.obs[i, 1] = y_max
    #                 self.obs[i, 4] = -abs(self.obs[i, 4])  # flip to negative
    #             elif self.obs[i, 1] <= y_min:
    #                 self.obs[i, 1] = y_min
    #                 self.obs[i, 4] = abs(self.obs[i, 4])   # flip to positive
    
    def set_obs_meta(self, meta_list):
        """
        meta_list: 길이 N인 리스트.
        각 원소는 dict 예) {'mode':0/1, 'v_max':0.35, 'theta':초기각(라드)}
        - mode=0: 등속 (기존과 동일)
        - mode=1: 랜덤워커(방향 랜덤 요동)
        """
        if not isinstance(meta_list, list):
            raise ValueError("meta_list must be a list")
        if len(meta_list) != len(self.obs):
            raise ValueError("meta_list length must match self.obs rows")
        self.obs_meta = meta_list
        
    def _ensure_obs_meta(self):
        """메타가 없으면 등속 기본값으로 자동 구성."""
        if self.obs_meta and len(self.obs_meta) == len(self.obs):
            return
        self.obs_meta = []
        for i in range(len(self.obs)):
            vx, vy = float(self.obs[i,3]), float(self.obs[i,4])
            vmag = float(np.hypot(vx, vy))
            theta0 = float(np.arctan2(vy, vx)) if vmag > 1e-9 else float(self._rng.uniform(-np.pi, np.pi))
            self.obs_meta.append({'mode': 0, 'v_max': vmag, 'theta': theta0})
        
    @staticmethod  
    def make_random_obstacles7(n_rand, v_obs_max, x_range, y_spawn_range, r_range, y_bounds, seed=42, rand_obs=True):
        if not rand_obs:
            return np.empty((0, 7), dtype=float) , []
        rng = np.random.default_rng(seed)
        x_min, x_max = x_range
        y_min_spawn, y_max_spawn = y_spawn_range
        r_min, r_max = r_range
        y_min_g, y_max_g = y_bounds

        rows = []
        metas = []
        for _ in range(n_rand):
            x0 = rng.uniform(x_min, x_max)
            y0 = rng.uniform(y_min_spawn, y_max_spawn)
            r  = rng.uniform(r_min, r_max)
            theta0 = rng.uniform(-np.pi, np.pi)
            if np.cos(theta0) >= 0.0:
                theta0 += np.pi
                if theta0 > np.pi:
                    theta0 -= 2*np.pi
            vx0, vy0 = v_obs_max*np.cos(theta0), v_obs_max*np.sin(theta0)
            rows.append([x0, y0, r, vx0, vy0, y_min_g, y_max_g])
            metas.append({'mode': 1, 'v_max': v_obs_max, 'theta': theta0})
        return np.array(rows, dtype=float), metas
    
    def step_dyn_obs(self):
        """
        self.obs: (N,7) = [x, y, r, vx, vy, y_min, y_max]
        self.obs_meta[i] = {'mode':0/1, 'v_max':..., 'theta':...}
        - mode=0: constant velocity
        - mode=1: random agents
        """
        if len(self.obs) == 0:
            return

        if not (isinstance(self.obs, np.ndarray) and self.obs.ndim == 2 and self.obs.shape[1] == 7):
            self.obs = np.array(self.obs, dtype=float).reshape(-1, 7)

        # 메타 없으면 기본값 생성
        self._ensure_obs_meta()

        for i in range(self.obs.shape[0]):
            x, y, r, vx, vy, y_min, y_max = self.obs[i, :7]
            meta = self.obs_meta[i]
            mode   = int(meta.get('mode', 0))
            v_max  = float(meta.get('v_max', np.hypot(vx, vy)))
            theta  = float(meta.get('theta', np.arctan2(vy, vx) if v_max>1e-9 else 0.0))

            if mode == 1:
                # --- 랜덤워커: 방향에 소음 + 간헐적 큰 턴 ---
                dtheta = self._rng.normal(0.0, 0.0)          # 작은 방향 요동
                if self._rng.random() < 0.05:                 # 5% 확률 큰 턴
                    dtheta += self._rng.normal(0.0, 0.2)
                theta += dtheta
                vx, vy = v_max * np.cos(theta), v_max * np.sin(theta)
                meta['theta'] = theta   # 메타에 최신 각도 저장

            # 위치 업데이트
            x_new = x + vx * self.dt
            y_new = y + vy * self.dt

            # y 경계 반사 처리
            if y_new >= y_max:
                y_new = y_max
                vy = -abs(vy)
                if mode == 1:
                    meta['theta'] = -meta['theta']            # x축 대칭 반사
                    vx, vy = v_max*np.cos(meta['theta']), v_max*np.sin(meta['theta'])
            elif y_new <= y_min:
                y_new = y_min
                vy =  abs(vy)
                if mode == 1:
                    meta['theta'] = -meta['theta']
                    vx, vy = v_max*np.cos(meta['theta']), v_max*np.sin(meta['theta'])

            # 쓰기
            self.obs[i, 0] = x_new
            self.obs[i, 1] = y_new
            self.obs[i, 3] = vx
            self.obs[i, 4] = vy
        
    def render_dyn_obs(self):
        if len(self.obs_vel_arrows) != len(self.obs):
            for arrow in self.obs_vel_arrows:
                arrow.remove()
            self.obs_vel_arrows = []
            
            if self.obs.shape[0] > 0 and self.obs.shape[1] >= 5:
                for _ in range(len(self.obs)):
                    arrow = patches.Arrow(0, 0, 0, 0, width=0.2, color='orange', zorder=5)
                    self.ax.add_patch(arrow)
                    self.obs_vel_arrows.append(arrow)

        for i, obs_info in enumerate(self.obs):
            # obs: [x, y, r, vx, vy]
            ox, oy, r = obs_info[:3]
            self.dyn_obs_patch[i].center = ox, oy
            self.dyn_obs_patch[i].set_radius(r)

            # Check if there are arrows to update
            if i < len(self.obs_vel_arrows):
                vx, vy = obs_info[3], obs_info[4]
                
                # Remove the old arrow and add a new one to update its properties
                # This is a robust way to handle patches in matplotlib animations
                self.obs_vel_arrows[i].remove()
                
                # You can scale the vector length for better visualization, e.g., multiply by 0.5
                arrow_scale = 1.0 
                new_arrow = patches.Arrow(ox, oy, vx * arrow_scale, vy * arrow_scale, 
                                          width=0.2, color='orange', zorder=5)
                
                self.ax.add_patch(new_arrow)
                self.obs_vel_arrows[i] = new_arrow

    def draw_plot(self, pause=0.01, force_save=False):
        if self.show_animation:
            if self.dyn_obs_patch is None:
                # Initialize moving obstacles
                self.dyn_obs_patch = [self.ax.add_patch(plt.Circle(
                    (0, 0), 0, edgecolor='black', facecolor='gray', fill=True)) for _ in range(len(self.obs))]
                self.init_obs_info = self.obs.copy()
                
            self.render_dyn_obs()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # move the square frame of the plot based on robot's x position
            # if self.robot_spec['model'] in ['VTOL2D']:
            #     x = np.clip(self.robot.X[0, 0], 7.5, 67.5)
            #     self.ax.set_xlim(x-7.5, x+7.5)
            #     self.ax.set_ylim(0, 15)
            #     self.fig.tight_layout()
                
            plt.pause(pause)
            if self.save_animation:
                self.ani_idx += 1
                if force_save or self.ani_idx % self.save_per_frame == 0:
                    plt.savefig(self.current_directory_path +
                                "/output/animations/" + "t_step_" + str(self.ani_idx//self.save_per_frame).zfill(4) + ".png", dpi=300)
                    # plt.savefig(self.current_directory_path +
                    #             "/output/animations/" + "t_step_" + str(self.ani_idx//self.save_per_frame).zfill(4) + ".svg")

    def control_step(self):
        '''
        Simulate one step of tracking control with CBF-QP with the given waypoints.
        Output: 
            - -2 or QPError: if the QP is infeasible or the robot collides with the obstacle
            - -1: all waypoints reached
            - 0: normal
            - 1: visibility violation
        '''
        # update state machine
        if self.state_machine == 'stop':
            if self.robot.has_stopped():
                if self.enable_rotation:
                    self.state_machine = 'rotate'
                else:
                    self.state_machine = 'track'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        # 1. Update the detected obstacles
        detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)
        # self.nearest_obs = self.get_nearest_obs(detected_obs)
        self.nearest_multi_obs = self.get_nearest_unpassed_obs(detected_obs, obs_num=self.num_constraints)
        if self.nearest_multi_obs is not None:
            self.nearest_obs = self.nearest_multi_obs[0].reshape(-1, 1)

        # 2. Update Moving Obstacles
        self.step_dyn_obs()

        # 3. Compuite nominal control input, pre-defined in the robot class
        if self.state_machine == 'rotate':
            goal_angle = np.arctan2(self.goal[1] - self.robot.X[1, 0],
                                    self.goal[0] - self.robot.X[0, 0])
            if self.robot_spec['model'] in ['SingleIntegrator2D', 'DoubleIntegrator2D']:
                self.u_att = self.robot.rotate_to(goal_angle)
                u_ref = self.robot.stop()
            elif self.robot_spec['model'] in ['Unicycle2D', 'DynamicUnicycle2D', 'KinematicBicycle2D', 'KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF', 'Quad2D', 'VTOL2D']:
                u_ref = self.robot.rotate_to(goal_angle)
        elif self.goal is None:
            u_ref = self.robot.stop()
        else:
            # Normal waypoint tracking
            if self.pos_controller_type == 'optimal_decay_cbf_qp':
                u_ref = self.robot.nominal_input(self.goal, k_omega=3.0, k_a=0.5, k_v=0.5)
            else:
                u_ref = self.robot.nominal_input(self.goal)

        # 4. Update the CBF constraints & 5. Solve the control problem
        control_ref = {'state_machine': self.state_machine,
                       'u_ref': u_ref,
                       'goal': self.goal}
        
        if self.pos_controller_type in ['optimal_decay_cbf_qp', 'cbf_qp']:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_multi_obs) 
        else:
            u = self.pos_controller.solve_control_problem(
                self.robot.X, control_ref, self.nearest_multi_obs)

        # Guard against None/NaN control output; mark infeasible so tracking stops.
        try:
            invalid_u = (u is None) or (not np.all(np.isfinite(u)))
        except Exception:
            invalid_u = True
        if invalid_u:
            try:
                self.pos_controller.status = 'infeasible'
            except Exception:
                pass
            u = self.robot.stop()

        plt.figure(self.fig.number)

        # 6. Draw collision cones/parabolas for C3BF/DPCBF
        if self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            self.robot.draw_collision_cone(self.robot.X, self.nearest_multi_obs, self.ax)
        elif self.robot_spec['model'] == 'KinematicBicycle2D_DPCBF':
            self.robot.draw_collision_parabola(self.robot.X, self.nearest_multi_obs, self.ax) 

        # 7. Update the attitude controller
        if self.state_machine == 'track' and self.att_controller is not None:
            # att_controller is only defined for integrators
            self.u_att = self.att_controller.solve_control_problem(
                    self.robot.X, self.robot.yaw, u)

        # 8. Raise an error if the QP is infeasible, or the robot collides with the obstacle
        collide = self.is_collide_unknown()
        
        if self.pos_controller.status != 'optimal' or collide:
            cause = "Collision" if collide else "Infeasible"
            self.draw_infeasible()
            print(f"{cause} detected !!")
            if self.raise_error:
                raise InfeasibleError(f"{cause} detected !!")
            return -2

        # 9. Step the robot
        self.robot.step(u, self.u_att)
        self.u_pos = u

        if hasattr(self.robot, "update_occlusion_polygons") and \
           hasattr(self.pos_controller, "occlusion_scenarios"):

            kappa = getattr(self.pos_controller, "kappa", 10.0)

            self.robot.update_occlusion_polygons(
                self.pos_controller.occlusion_scenarios,
                kappa=kappa,
                show_true_occ=True,
                show_true_occ_T=True,
                show_softmax_occ_T=False,
                T_rollout=getattr(self.pos_controller, "T_horizon", 3.0),
                grid_res=0.05,
            )

        if self.show_animation:
            self.robot.render_plot()

        # 10. Update sensing information
        if 'sensor' in self.robot_spec and self.robot_spec['sensor'] == 'rgbd':
            self.robot.update_sensing_footprints()
            self.robot.update_safety_area()

            beyond_flag = self.robot.is_beyond_sensing_footprints()
            if beyond_flag and self.show_animation:
                pass
                # print("Visibility Violation")
        else:
            beyond_flag = 0 # not checking sensing footprint

        if self.goal is None and self.state_machine != 'stop':
            return -1  # all waypoints reached
        return beyond_flag

def single_agent_main(controller_type):
    dt = 0.05
    model = 'DoubleIntegrator2D' # SingleIntegrator2D, DoubleIntegrator2D, DynamicUnicycle2D, KinematicBicycle2D, KinematicBicycle2D_C3BF, KinematicBicycle2D_DPCBF, Quad2D

    waypoints = [
         [1, 7.5, 0],
         [20, 7.5, 0],
    ]

    # Define dynamic obs
    # known_obs = np.array([
    #     [15.0, 12.3, 0.5],  # obstacle 1
    #     # [8.0, 11.0, 0.5],  # obstacle 3
    #     # [10.0, 5.0, 0.5],  # obstacle 5
    #     # [12.0, 7.0, 0.5],  # obstacle 7
    #     [16.0, 6.5, 0.5],  # obstacle 9
    #     [17.0, 7.0, 0.5],  # obstacle 11
    #     [18.0, 7.5, 0.5],  # obstacle 13
    #     [22.0, 12.0, 0.5],  # obstacle 15
    # ])
    # Bus scenario
    # known_obs = np.array([
    #     [7.0, 6.0, 0.5],  # obstacle 3
    #     [8.0, 6.0, 0.5],  # obstacle 5
    #     [7.0, 5.5, 0.5],  # obstacle 7
    #     [8.0, 5.5, 0.5],  # obstacle 9
    #     [7.0, 5.0, 0.5],  # obstacle 11
    #     [8.0, 5.0, 0.5],  # obstacle 13
    #     [7.0, 4.5, 0.5],  # obstacle 11
    #     [8.0, 4.5, 0.5],  # obstacle 13
    #     # [22.0, 12.0, 0.5],  # obstacle 15
    # ])
    # Supermarket Scenario
    known_obs = np.array([
        [8.0, 5.0, 0.5],  # obstacle 1
        [10.0, 7.0, 0.5],  # obstacle 2
        [12.0, 11.0, 0.5],  # obstacle 3
        [14.0, 6.5, 0.5],  # obstacle 4
        [16.0, 3.0, 0.5],  # obstacle 5
        [18.0, 7.5, 0.5],  # obstacle 6
        [20.0, 8.9, 0.5],  # obstacle 6
        [22.0, 10.6, 0.5],  # obstacle 6
        [24.0, 12.0, 0.5],  # obstacle 7
    ])
    # LoS scenario static/dyn
    # known_obs = np.array([
    #     [15.0, 7.5, 0.5],  # obstacle 1
    # ])
    # Crowd Scenario
    # known_obs = np.array([     
    #     [8.0, 1.5, 0.3],    # obstacle 2
    #     [9.0, 7.8, 0.3],    # obstacle 3
    #     [10.0, 3.2, 0.3],   # obstacle 4
    #     [11.0, 11.9, 0.3],  # obstacle 5
    #     [12.0, 9.1, 0.3],   # obstacle 6
    #     [13.0, 2.8, 0.3],   # obstacle 7
    #     [14.0, 12.3, 0.3],  # obstacle 2
    #     [15.0, 4.7, 0.3],   # obstacle 3
    #     [16.0, 10.6, 0.3],  # obstacle 4
    #     [17.0, 8.0, 0.3],   # obstacle 5
    #     [18.0, 5.4, 0.3],   # obstacle 6
    #     [19.0, 13.0, 0.3],  # obstacle 7
    #     [20.0, 6.3, 0.3],   
    #     [21.0, 8.9, 0.3],
    #     [22.0, 5.4, 0.3],    # obstacle 1
    # ])
    # known_obs = np.array([     
    #     [8.0, 1.5, 0.3],    # obstacle 2
    #     [9.0, 7.8, 0.3],    # obstacle 3
    #     [10.0, 3.2, 0.3],   # obstacle 4
    #     [11.0, 11.9, 0.3],  # obstacle 5
    #     # [12.0, 9.1, 0.3],   # obstacle 6
    #     [13.0, 2.8, 0.3],   # obstacle 7
    #     [14.0, 12.3, 0.3],  # obstacle 2
    #     [15.0, 4.7, 0.3],   # obstacle 3
    #     # [16.0, 10.6, 0.3],  # obstacle 4
    #     [17.0, 8.0, 0.3],   # obstacle 5
    #     [18.0, 5.4, 0.3],   # obstacle 6
    #     [19.0, 10.0, 0.3],  # obstacle 7
    #     [20.0, 6.3, 0.3],   
    #     [21.0, 8.9, 0.3],
    #     [22.0, 5.4, 0.3],    # obstacle 1
    # ])
    # Appear Unknown obs in LoS
    # known_obs = np.array([
    #     #[7.0, 6.0, 0.5],  # obstacle 3
    #     #[8.0, 6.0, 0.5],  # obstacle 5
    #     #[7.0, 5.5, 0.5],  # obstacle 7
    #     [9.0, 9.0, 0.5],  # obstacle 9
    #     #[1.0, 5.0, 0.5],  # obstacle 11
    #     [10.0, 10.0, 0.5],  # obstacle 13
    #     # [22.0, 12.0, 0.5],  # obstacle 15
    # ])
    # wall w/ straight dyn obs
    # known_obs = np.array([
    #     [12.0, 5.0, 0.6],  # obstacle 3
    #     [13.0, 5.5, 0.6],  # obstacle 5
    #     [14.0, 6.0, 0.6],  # obstacle 7
    #     [15.0, 6.5, 0.6],  # obstacle 9
    #     [15.5, 2.0, 0.5],  # obstacle 11
    #     # [2.0, 5.0, 0.5],  # obstacle 13
    #     # [22.0, 12.0, 0.5],  # obstacle 15
    # ])

    RAND_OBS_ENABLE = True
    dynamic_obs = []  
    for i, obs_info in enumerate(known_obs):
        ox, oy, r = obs_info[:3]
        if i % 2 == 0:
            vx, vy = -0.15, -0.15
        else:
            vx, vy = -0.15, 0.15
        y_min, y_max = 1.0, 14.0
        # if i <= 3:
        #     vx, vy = 0.0, 0.0
        # else:
        #     vx, vy = -0.0, 0.5
        dynamic_obs.append([ox, oy, r, vx, vy, y_min, y_max])
    known_obs = np.array(dynamic_obs, dtype=float)

    rand_rows, rand_meta = LocalTrackingControllerDyn.make_random_obstacles7(
        n_rand=10,
        v_obs_max=0.5,
        x_range=(15.0, 25.0),
        y_spawn_range=(0.0, 15.0),
        r_range=(0.2, 0.25),
        y_bounds=(0.0, 15.0),
        seed=42,
        rand_obs= RAND_OBS_ENABLE,
    )
            
    if rand_rows.size:
        known_obs = np.vstack([known_obs, rand_rows])

    env_width = 24.0
    env_height = 15.0
    if model == 'SingleIntegrator2D':
        robot_spec = {
            'model': 'SingleIntegrator2D',
            'v_max': 1.0,
            'radius': 0.25
        }
    elif model == 'DoubleIntegrator2D':
        robot_spec = {
            'model': 'DoubleIntegrator2D',
            'v_max': 1.0,
            'a_max': 1.0,
            'radius': 0.25,
            'debug_backup_qp': True,
            'sensing_range': 10.0
        }
    elif model == 'DynamicUnicycle2D':
        robot_spec = {
            'model': 'DynamicUnicycle2D',
            'w_max': 0.5,
            'a_max': 0.5,
            'sensor': 'rgbd',
            'radius': 0.25
        }
    elif model == 'KinematicBicycle2D':
        robot_spec = {
            'model': 'KinematicBicycle2D',
            'a_max': 0.5,
            'sensor': 'rgbd',
            'radius': 0.5
        }
    elif model == 'KinematicBicycle2D_C3BF':
        robot_spec = {
            'model': 'KinematicBicycle2D_C3BF',
            'a_max': 5.0,
            'radius': 0.3
        }
    elif model == 'KinematicBicycle2D_DPCBF':
        robot_spec = {
            'model': 'KinematicBicycle2D_DPCBF',
            'a_max': 5.0,
            # 'sensor': 'rgbd',
            'radius': 0.3
        }
    elif model == 'Quad2D':
        robot_spec = {
            'model': 'Quad2D',
            'f_min': 3.0,
            'f_max': 10.0,
            'sensor': 'rgbd',
            'radius': 0.25
        }
        # override the waypoints with z axis
        waypoints = [
            [2, 2, 0, math.pi/2],
            [2, 12, 1, 0],
            [12, 12, -1, 0],
            [12, 2, 0, 0]
        ]

    waypoints = np.array(waypoints, dtype=np.float64)

    if model in ['SingleIntegrator2D', 'DoubleIntegrator2D', 'Quad2D']:
        x_init = waypoints[0]
    else:
        x_init = np.append(waypoints[0], 1.0)
    
    if known_obs.shape[1] != 7:
        known_obs = np.hstack((known_obs, np.zeros((known_obs.shape[0], 2)))) # Set static obs velocity 0.0 at (5, 5)
    
    plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
    ax, fig = plot_handler.plot_grid("") # you can set the title of the plot here
    env_handler = env.Env()

    tracking_controller = LocalTrackingControllerDyn(x_init, robot_spec,
                                                  controller_type=controller_type,
                                                  dt=dt,
                                                  show_animation=True,
                                                  save_animation=False,
                                                  show_mpc_traj=False,
                                                  ax=ax, fig=fig,
                                                  env=env_handler)

    # Set obstacles
    tracking_controller.obs = known_obs.astype(float)
    N_const = known_obs.shape[0] - rand_rows.shape[0]
    const_meta = []
    for row in known_obs[:N_const]:
        vx, vy = float(row[3]), float(row[4])
        vmag = float(np.hypot(vx, vy))
        theta0 = float(np.arctan2(vy, vx)) if vmag > 1e-9 else 0.0
        const_meta.append({'mode': 0, 'v_max': vmag, 'theta': theta0})

    meta = const_meta + rand_meta
    tracking_controller.set_obs_meta(meta)
    # tracking_controller.set_unknown_obs(unknown_obs)
    tracking_controller.set_waypoints(waypoints)
    unexpected_beh = tracking_controller.run_all_steps(tf=300)

if __name__ == "__main__":
    from utils import plotting
    from utils import env
    import math

    # single_agent_main(controller_type={'pos': 'cbf_qp'})
    # single_agent_main(controller_type={'pos': 'mpc_cbf'})
    # single_agent_main(controller_type={'pos': 'mpc_cbf', 'att': 'gatekeeper'}) # only Integrators have attitude controller, otherwise ignored
    single_agent_main(controller_type={'pos': 'backup_cbf_qp'})