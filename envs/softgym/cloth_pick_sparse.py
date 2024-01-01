import numpy as np
# import random
from math import floor
import pickle
import os.path as osp
import os
import pyflex
import cv2
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
import copy
from copy import deepcopy
from gym import spaces
import time
import datetime
import pdb


class ClothPickSparseEnv(ClothEnv):
    def __init__(self, reward_type = "sparse", cached_states_path='cloth_drop_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        self.vertical_group_a = self.flat_group_b = None
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        assert self.action_tool.num_picker == 2  # Two drop points for this task
        self.prev_dist = None  # Should not be used until initialized

        self._num_key_points = 16
        self.dist_thresh = 0.02
        self.reward_type = reward_type
        self.obs_img_size = 128

        # disable the patch_deprecated_methods during registration
        self._gym_disable_underscore_compat = True

        self._goal_save_dir = "save/cloth_pick/goals/"
        if not osp.exists("save/cloth_pick/goals/"):
            os.makedirs("save/cloth_pick/goals/")

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

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [80, 58],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0.97199, 0.94942, 1.35691]),
                                   'angle': np.array([0.633549, -0.497932, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0
        }
        return config

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = x_low
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        return curr_pos

    def _set_to_vertical(self, x_low, height_low):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < 0.5:
            curr_pos[:, 1] += 0.5 - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_flat_pos(self):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)
        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _sample_cloth_size(self):
        # return np.random.randint(60, 100), np.random.randint(60, 100)
        return self.np_random.randint(60, 100), np.random.randint(60, 100)

    def generate_env_variation(self, num_variations=1, vary_cloth_size=False):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.1  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

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

            # pickpoints = self._get_drop_point_idx()[:2]  # Pick two corners of the cloth and wait until stablize

            # # config['target_pos'] = self._get_flat_pos()
            # self.x_low = np.random.random() * 0.2 - 0.1
            self.x_low = self.np_random.random() * 0.2 - 0.1
            # self._set_to_vertical(self.x_low, height_low=np.random.random() * 0.1 + 0.1)

            # # Get height of the cloth without the gravity. With gravity, it will be longer
            # # p1, _, p2, _ = self._get_key_point_idx() 

            # curr_pos = pyflex.get_positions().reshape(-1, 4)
            # curr_pos[0] += np.random.random() * 0.001  # Add small jittering
            # original_inv_mass = curr_pos[pickpoints, 3]
            # curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            # pickpoint_pos = curr_pos[pickpoints, :3]
            # pyflex.set_positions(curr_pos.flatten())

            # picker_radius = self.action_tool.picker_radius
            # self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
            # self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))

            # # Pick up the cloth and wait to stablize
            # for j in range(0, max_wait_step):
            #     pyflex.step()
            #     curr_pos = pyflex.get_positions().reshape((-1, 4))
            #     curr_vel = pyflex.get_velocities().reshape((-1, 3))
            #     if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
            #         break
            #     curr_pos[pickpoints, :3] = pickpoint_pos
            #     pyflex.set_positions(curr_pos)
            # curr_pos = pyflex.get_positions().reshape((-1, 4))
            # curr_pos[pickpoints, 3] = original_inv_mass
            # pyflex.set_positions(curr_pos.flatten())
            # generated_configs.append(deepcopy(config))
            # print('config {}: {}'.format(i, config['camera_params']))
            # generated_states.append(deepcopy(self.get_state()))

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

    def compute_reward(self, achieved_goal, desired_goal, info):
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
        if self.observation_mode == 'cam_rgb':
            obs, achieved_goal = self._reset()
        elif self.observation_mode == 'key_point':
            obs = self._reset()
            achieved_goal = obs.reshape(-1, 3)[:self._num_key_points].flatten()

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

    def step(self, action):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        # process action
        # For simplicity, holding the corner all time.
        # action = action.reshape(-1,4)
        # action[:, 3] = 1
        # action = action.reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
       
        for i in range(self.action_repeat):
            self._step(action)

        if self.observation_mode == 'cam_rgb':
            obs, achieved_goal = self._get_obs()
            achieved_goal = self._normalize_points(achieved_goal)
        elif self.observation_mode == 'key_point':
            obs = self._get_obs()
            achieved_goal = obs.copy().reshape(-1, 3)[:self._num_key_points].flatten()
            achieved_goal = self._normalize_points(achieved_goal)
            obs = self._normalize_points(obs)

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
        width, height = self.camera_params['default_camera']['width'], self.camera_params['default_camera']['height']
        img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
        img = img[int(0.25 * height):int(0.75 * height), int(0.25 * width):int(0.75 * width)]
        img = cv2.resize(img.astype(np.uint8), (target_w, target_h))
        return img

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            img = pyflex.render()
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

    # def _get_key_point_idx(self, x_split):
    #     indices = [0]
    #     dimx, dimy = self.current_config['ClothSize']
    #     idx_p1 = 0
    #     idx_p2 = dimx*(dimy - 1)
    #     idx_p3 = x_split 
    #     idx_p4 = dimx*(dimy-1) + x_split
    #     idx_p5 = dimx-1
    #     idx_p6 = dimx*dimy - 1
    #     return np.array([idx_p1, idx_p2, idx_p3, idx_p4, idx_p5, idx_p6])

    def _get_key_point_idx(self):
        dimx, dimy = self.current_config['ClothSize']
        low, high = 0, floor(dimx *0.6)
        interval = (high - low) // 6
        idx_p1 = 0
        idx_p2 = dimx*(dimy - 1)

        idx_p3 = interval
        idx_p5 = 2*interval
        idx_p7 = 3*interval
        idx_p9 = 4*interval
        idx_p11 = 5*interval
        idx_p13 = high

        idx_p4 = dimx*(dimy-1)+ interval
        idx_p6 = dimx*(dimy-1) + 2*interval
        idx_p8 = dimx*(dimy-1) + 3*interval
        idx_p10 = dimx*(dimy-1) + 4*interval
        idx_p12 = dimx*(dimy-1) + 5*interval
        idx_p14 = dimx*(dimy-1) + high

        idx_p15 = dimx-1
        idx_p16 = dimx*dimy - 1
        return np.array([idx_p1, idx_p2, idx_p3, idx_p4, idx_p5, idx_p6, idx_p7, idx_p8,\
             idx_p9, idx_p10, idx_p11, idx_p12, idx_p13, idx_p14, idx_p15, idx_p16])

    def _normalize_points(self, points):
        input_shape = points.shape
        pos = pyflex.get_positions().reshape(-1, 4)
        points = points.reshape(-1 ,3)
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
        cloth_dimy = self.current_config['ClothSize'][1]
        # self.x_split = np.random.randint(floor(cloth_dimx *0.3), floor(cloth_dimx *0.6))
        self.x_split = self.np_random.randint(floor(cloth_dimx *0.3), floor(cloth_dimx *0.6))
        self.key_point_indices = self._get_key_point_idx()
        self.vertical_group_a = particle_grid_idx[:, :self.x_split].flatten()
        self.flat_group_b = particle_grid_idx[:, self.x_split:].flatten()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        x = np.array([i * self.cloth_particle_radius for i in range(cloth_dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(cloth_dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)
        goal_pos = np.zeros([cloth_dimx * cloth_dimy, 3], dtype=np.float32)
        goal_pos[:, 0] = xx.flatten()
        goal_pos[:, 2] = yy.flatten()
        goal_pos[:, 1] = 5e-3
        # set group_a vertical
        x_offset = goal_pos[cloth_dimx - 1, 0] - self.x_low
        goal_pos [:,0] -= x_offset
        
        # goal_pos[self.vertical_group_a, 0] = goal_pos[self.vertical_group_a[-1]+1,0]
        # print(self.vertical_group_a[-1])
        # print(cloth_dimx*(cloth_dimy-1) + self.x_split)
        goal_pos[self.vertical_group_a, 0] = goal_pos[cloth_dimx*(cloth_dimy-1) + self.x_split ,0]
        for j in range(cloth_dimy):
            for i in range(self.x_split):
                goal_pos[j*cloth_dimx+i,1] = 5e-3 + (self.x_split-i)*self.cloth_particle_radius
    
        self.current_config['goal_pos'] = goal_pos
        curr_pos[:,:3] = goal_pos
        pyflex.set_positions(curr_pos.flatten())
        pyflex.set_velocities(np.zeros_like(goal_pos.flatten()))
        # pyflex.step()

        center_object()

        colors = np.zeros(num_particles)
        colors[self.vertical_group_a] = 1
      
        # read goal state
        # particle_pos = curr_pos[:, :3]
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self.key_point_indices, :3]
        goal = self._normalize_points(keypoint_pos.flatten())

        # visualize the goal scene
        self.goal_img = self.render_goal(200, 200)
        return goal

    def _reset(self):
        """ Right now only use one initial state"""
        # self.prev_dist = self._get_current_dist(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.0, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))
            # self.action_tool.visualize_picker_boundary()
        # self.performance_init = None
        # info = self._get_info()
        # self.performance_init = info['performance']
        return self._get_obs()

    def _step(self, action):
        self.action_tool.step(action)
        pyflex.step()
        return

    def _get_current_dist(self, pos):
        goal_pos = self.get_current_config()['goal_pos']
        curr_pos = pos.reshape((-1, 4))[:, :3]
        curr_dist = np.mean(np.linalg.norm(curr_pos - goal_pos, axis=1))
        return curr_dist

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
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

    # def compute_reward(self, action=None, obs=None, set_prev_reward=True):
    #     particle_pos = pyflex.get_positions()
    #     curr_dist = self._get_current_dist(particle_pos)
    #     r = - curr_dist
    #     return r

    # def _get_info(self):
    #     particle_pos = pyflex.get_positions()
    #     curr_dist = self._get_current_dist(particle_pos)
    #     performance = -curr_dist
    #     performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
    #     return {
    #         'performance': performance,
    #         'normalized_performance': (performance - performance_init) / (0. - performance_init)}
