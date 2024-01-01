import numpy as np
import cv2
import os
import os.path as osp
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
from gym import spaces
import time
import datetime
import pdb

class ClothFoldSparseEnv(ClothEnv):
    def __init__(self, reward_type = "sparse", cached_states_path='cloth_fold_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        self._num_key_points = 16
        self.dist_thresh = 0.02
        self.reward_type = reward_type
        self.obs_img_size = 200

        # disable the patch_deprecated_methods during registration
        self._gym_disable_underscore_compat = True

        self._goal_save_dir = "save/cloth_fold/goals/"
        if not osp.exists("save/cloth_fold/goals/"):
            os.makedirs("save/cloth_fold/goals/")

        obs = self.reset()
        if self.observation_mode == 'key_point':
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            ))
        elif self.observation_mode == 'cam_rgb':
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_img_size, self.obs_img_size, 3),
                                                dtype=np.float32),
            ))

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def generate_env_variation(self, num_variations=2, vary_cloth_size=False):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            if self.action_mode in ['sawyer', 'franka']: # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for wait_i in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object()
            # angle = (np.random.random() - 0.5) * np.pi / 2
            # self.rotate_particles(angle)

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        # rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_colors = self.np_random.randint(0, 5, size=rand_size)
        # rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        rand_index = self.np_random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,\
                p13, p14, p15, p16 = self._get_key_point_idx()
            key_point_pos = particle_pos[(p1, p2), :3] # Was changed from from p1, p4.
            middle_point = np.mean(key_point_pos, axis=0)
            self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.0, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=key_point_pos + np.array([0., picker_radius, 0.]))

            # picker_low = self.action_tool.picker_low
            # picker_high = self.action_tool.picker_high
            # offset_x = self.action_tool._get_pos()[0][0][0] - picker_low[0] - 0.3
            # picker_low[0] += offset_x
            # picker_high[0] += offset_x
            # picker_high[0] += 1.0
            # self.action_tool.update_picker_boundary(picker_low, picker_high)

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        # self.set_colors(colors) # TODO the phase actually changes the cloth dynamics so we do not change them for now. Maybe delete this later.

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Reward is the distance between achieved_goal and desired_goal
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        _shape = achieved_goal.shape[:-1] + (self._num_key_points, 3)
        achieved_goal = achieved_goal.reshape(_shape)
        desired_goal = desired_goal.reshape(_shape)
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1).max(-1)
        if self.reward_type == "sparse":
            return - (dist >= self.dist_thresh).astype(np.float32)
        else:
            return - dist

    def reset(self):
        self.goal = self._sample_goal()

        self.current_config = self.cached_configs[0]
        self.set_scene(self.cached_configs[0], self.cached_init_states[0])
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.time_step = 0
        # obs = self._reset()
        # achieved_goal = obs.reshape(-1, 3)[:self._num_key_points].flatten()

        if self.observation_mode == 'cam_rgb':
            obs = self._reset()
            achieved_goal = obs.copy()
        elif self.observation_mode == 'key_point':
            obs = self._reset()
            achieved_goal = obs.reshape(-1, 3)[:self._num_key_points].flatten()
        cv2.imwrite(self._goal_save_dir + "cloth_fold_gpal.png", self.goal_img)
        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

    def step(self, action):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        # process action
        action = action.reshape(-1,4)
        action[:, 3] = 1
        action = action.reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i in range(self.action_repeat):
            self._step(action)

        if self.observation_mode == 'cam_rgb':
            obs = self._get_obs()
            achieved_goal = obs.copy()
            #achieved_goal = self._normalize_points(achieved_goal)

        elif self.observation_mode == 'key_point':
            obs = self._get_obs()
            achieved_goal = obs.copy().reshape(-1, 3)[:self._num_key_points].flatten()
            achieved_goal = self._normalize_points(achieved_goal)
            obs = self._normalize_points(obs)

        # obs = self._get_obs()
        # achived_goal = obs.copy().reshape(-1, 3)[:self._num_key_points].flatten()
        # achived_goal = self._normalize_points(achived_goal)
        # obs = self._normalize_points(obs)

        desired_goal = self.goal
        reward = self.compute_reward(achieved_goal, desired_goal, None)

        obs = {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal.copy()
        }
        info = {'is_success': reward >= - self.dist_thresh}

        self.time_step += 1

        done = False
        if self.time_step >= self.horizon:
            done = True
        return obs, reward, done, info

    def render_goal(self, target_w, target_h):
        img = pyflex.render()
        # img = pyflex.render_cloth()
        width, height = self.camera_params['default_camera']['width'], self.camera_params['default_camera']['height']
        img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
        img = img[int(0.125 * height):int(0.875 * height), int(0.125 * width):int(0.875 * width)]
        img = cv2.resize(img.astype(np.uint8), (target_w, target_h))
        return img

    def render(self, mode='rgb_array'): #for visualization
        if mode == 'rgb_array':
            img = pyflex.render()
            # img = pyflex.render_cloth()
            width, height = self.camera_params['default_camera']['width'], self.camera_params['default_camera']['height']
            img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
            goal_img = self.goal_img.copy()
            # attach goal patch on the rendered image
            goal_img[:10, :, :] = 0
            goal_img[:, :10, :] = 0
            goal_img[-10:, :, :] = 0
            goal_img[:, -10:, :] = 0
            img[30:230, 30:230] = goal_img
            return img
        elif mode == 'human':
            raise NotImplementedError

    # def get_image_with_goal(self, width=720, height=720): #for visualization
    #     """ use pyflex.render to get a rendered image. """
    #     img = self.render_with_goal(mode='rgb_array')
    #     img = img.astype(np.uint8)
    #     if width != img.shape[0] or height != img.shape[1]:
    #         img = cv2.resize(img, (width, height))
    #     return img
    
    # def render(self, mode='rgb_array'):
    #     if mode == 'rgb_array':
    #         img = pyflex.render()
    #         # img = pyflex.render_cloth()
    #         width, height = self.camera_params['default_camera']['width'], self.camera_params['default_camera']['height']
    #         img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
    #         return img
    #     elif mode == 'human':
    #         raise NotImplementedError

    def _get_key_point_idx(self):
        dimx, dimy = self.current_config['ClothSize']
        low, high = dimx // 5, dimx // 2 + 1
        interval = (high - low) // 5
        idx_p1 = 0
        idx_p2 = dimx*(dimy - 1)

        idx_p3 = low -1
        idx_p5 = low + interval -1
        idx_p7 = low + 2*interval -1
        idx_p9 = low + 3*interval -1
        idx_p11 = low + 4*interval -1
        idx_p13 = high

        idx_p4 = dimx*(dimy-1)+ low - 1
        idx_p6 = dimx*(dimy-1) + low + interval -1
        idx_p8 = dimx*(dimy-1) + low + 2*interval -1
        idx_p10 = dimx*(dimy-1) + low + 3*interval -1
        idx_p12 = dimx*(dimy-1) + low + 4*interval -1
        idx_p14 = dimx*(dimy-1) + high

        idx_p15 = dimx-1
        idx_p16 = dimx*dimy - 1
        return np.array([idx_p1, idx_p2, idx_p3, idx_p4, idx_p5, idx_p6, idx_p7, idx_p8,\
             idx_p9, idx_p10, idx_p11, idx_p12, idx_p13, idx_p14, idx_p15, idx_p16])

    def _normalize_points(self, points):
        input_shape = points.shape
        pos = pyflex.get_positions().reshape(-1, 4)
        points = points.reshape(-1 ,3).astype('float32')
        points[:, [0, 2]] -= np.mean(pos[:, [0, 2]], axis=0, keepdims=True)
        return points.reshape(input_shape)

    def _sample_goal(self):
        # reset scene
        self.current_config = self.cached_configs[0]
        init_state = self.cached_init_states[0]
        self.set_scene(self.current_config, init_state)

        num_particles = np.prod(self.current_config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(self.current_config['ClothSize'][1], self.current_config['ClothSize'][0])  # Reversed index here

        cloth_dimx = self.current_config['ClothSize'][0]
        # self.x_split = np.random.randint(cloth_dimx // 5, cloth_dimx // 2 +1)
        self.x_split = self.np_random.randint(cloth_dimx // 5, cloth_dimx // 2 +1)
        self.fold_group_a = particle_grid_idx[:, :self.x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, cloth_dimx - 2*self.x_split:cloth_dimx -self.x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
      
        pyflex.step()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[self.fold_group_a, :] = curr_pos[self.fold_group_b, :].copy()
        curr_pos[self.fold_group_a, 1] += 0.05  # group a particle position made tcurr_pos[self.fold_group_b, 1] + 0.05e at top of group b position.
        # curr_pos[self.fold_group_a, 1] += self.cloth_particle_radius

        pyflex.set_positions(curr_pos.flatten())
        pyflex.set_velocities(np.zeros_like(curr_pos[:,:3]))
        for i in range(10):
            pyflex.step()
        center_object

        curr_pos = pyflex.get_positions().reshape((-1, 4))[:,:3]
        key_point = self._get_key_point_idx()
        goal = self._normalize_points(curr_pos[key_point,:3].flatten())
    

        # visualize the goal scene
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([10, 10, 10])
        self.goal_img = self.render_goal(self.obs_img_size, self.obs_img_size)
        self.goal_keypoint_pos = goal.copy() #for computing reward

        if self.observation_mode == 'key_point':
            return goal
        elif self.observation_mode == 'cam_rgb':
            return self.goal_img

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            self.current_keypoint_pos = particle_pos[self._get_key_point_idx(), :3] #for computing reward
            return self.get_image(self.obs_img_size, self.obs_img_size)
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
            pos = keypoint_pos

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos


    # def _set_to_folded(self):
    #     config = self.get_current_config()
    #     num_particles = np.prod(config['ClothSize'], dtype=int)
    #     particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

    #     cloth_dimx = config['ClothSize'][0]
    #     x_split = cloth_dimx // 2
    #     fold_group_a = particle_grid_idx[:, :x_split].flatten()
    #     fold_group_b = np.flip(particle_grid_idx, axis=1)[:, cloth_dimx - 2*x_split:cloth_dimx -x_split].flatten()

    #     curr_pos = pyflex.get_positions().reshape((-1, 4))
    #     curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
    #     curr_pos[fold_group_a, 1] += 0.05  # group a particle position made tcurr_pos[self.fold_group_b, 1] + 0.05e at top of group b position.

    #     pyflex.set_positions(curr_pos)
    #     for i in range(10):
    #         pyflex.step()

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        # particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
            cam_pos, cam_angle = np.array([-0.0, 0.82, 0.82]), np.array([0, -45 / 180. * np.pi, 0.])
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [80, 58],
            'ClothStiff': [0.8, 1, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0
        }

        return config

    def _get_info(self):
        return {}

    # For 
    def get_obs(self):
        obs = self._get_obs()
        achieved_goal = obs.copy().reshape(-1, 3)[:self._num_key_points].flatten()
        achieved_goal = self._normalize_points(achieved_goal)
        obs = self._normalize_points(obs)
        desired_goal = self.goal

        obs = {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": desired_goal.copy()
        }
        return obs

