import numpy as np
import pickle
import os
import os.path as osp
import cv2
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from copy import deepcopy
# from softgym.utils.pyflex_utils import random_pick_and_place, center_object
from softgym.utils.pyflex_utils import center_object
from gym import spaces
import time
import datetime
# import pdb

class RopePickSparseEnv(RopeNewEnv):
    def __init__(self, reward_type = "sparse", cached_states_path='rope_pick_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        kwargs["num_picker"] = 1
        super().__init__(**kwargs)
        self.prev_distance_diff = None
        self._num_key_points = 8
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        
        self.dist_thresh = 0.03
        self.reward_type = reward_type

        # disable the patch_deprecated_methods during registration
        self._gym_disable_underscore_compat = True
        self._rope_symmetry = False

        self._goal_save_dir = "save/rope_pick/goals/"
        if not osp.exists("save/rope_pick/goals/"):
            os.makedirs("save/rope_pick/goals/")

        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        if config is None:
            config = self.get_default_config()
            config['camera_params']['default_camera']['pos'] = np.array([0.97199, 0.94942, 1.35691])
            config['camera_params']['default_camera']['angle'] = np.array([0.633549, -0.497932, 0])
        default_config = config
        config = deepcopy(default_config)
        config['segment'] = 40
        self.set_scene(config)

        self.update_camera('default_camera', default_config['camera_params']['default_camera'])
        config['camera_params'] = deepcopy(self.camera_params)
        self.action_tool.reset([0., -1., 0.])

        # rope_particle_num = config['segment'] + 1
        # pickpoint = self._get_drop_point_idx(rope_particle_num, self._num_key_points)
        
        # curr_pos = pyflex.get_positions().reshape(-1, 4)
        # curr_pos[0] += np.random.random() * 0.001  # Add small jittering
        # pickpoint_pos = curr_pos[pickpoint, :3]
        # pyflex.set_positions(curr_pos.flatten())

        # picker_radius = self.action_tool.picker_radius
        # self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
        # self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))

        # self.random_pick_and_place(pick_num=4, pick_scale=0.005)
        center_object()
        generated_configs = deepcopy(config)
        print('config: {}'.format(config['camera_params']))
        generated_states = deepcopy(self.get_state())
        
        return [generated_configs] * num_variations, [generated_states] * num_variations

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Reward is the distance between the endpoints of the rope"""
        _shape = achieved_goal.shape[:-1] + (len(self.key_point_indices), 3)
        achieved_goal = achieved_goal.reshape(_shape)
        desired_goal = desired_goal.reshape(_shape)

        if self._rope_symmetry:
            # the symmetry of the rope
            dist1 = np.linalg.norm(achieved_goal - desired_goal, axis=-1).max(-1)
            dist2 = np.linalg.norm(np.flip(achieved_goal, axis=-2) - desired_goal, axis=-1).max(-1)
            dist = np.minimum(dist1, dist2)
        else:
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
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.reshape(-1, 4)
        # # action[:, 2] = 0
        # action[:, 3] = 1
        action = action.reshape(-1)
        for i in range(self.action_repeat):
            self._step(action)

        obs = self._get_obs()
        achived_goal = obs.copy().reshape(-1, 3)[:self._num_key_points].flatten()
        achived_goal = self._normalize_points(achived_goal)
        obs = self._normalize_points(obs)

        desired_goal = self.goal
        reward = self.compute_reward(achived_goal, desired_goal, None)

        obs = {
            "observation": obs.copy(),
            "achieved_goal": achived_goal.copy(),
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

    def _get_key_point_idx(self, num=None, key_point_num=4):
        indices = [0]
        n_mid_points = key_point_num - 2
        interval = (num - 2) // n_mid_points
        for i in range(1, 1 + n_mid_points):
            indices.append(i * interval)
        indices.append(num - 1)
        return indices

    def _normalize_points(self, points):
        input_shape = points.shape
        pos = pyflex.get_positions().reshape(-1, 4)
        points = points.reshape(-1 ,3)
        points[:, [0, 2]] -= np.mean(pos[:, [0, 2]], axis=0, keepdims=True)
        return points.reshape(input_shape)

    def _sample_goal(self):
        # reset scene
        config = self.cached_configs[0]
        init_state = self.cached_init_states[0]
        self.set_scene(config, init_state)
        rope_particle_num = config['segment'] + 1
        particle_idx = np.array(list(range(rope_particle_num)))
        self.key_point_indices = self._get_key_point_idx(rope_particle_num, self._num_key_points)

        # h = np.random.randint(int(rope_particle_num*0.3), int(rope_particle_num*0.6))
        h = self.np_random.randint(int(rope_particle_num*0.3), int(rope_particle_num*0.6))
        vertical_group_idx = particle_idx[:h]
        flat_group_idx = particle_idx[h:]

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        goal_pos = np.zeros([rope_particle_num, 3], dtype=np.float32)
        
        # goal_pos[vertical_group_idx, 1] = list(reversed([i*config['radius'] for i in range(h)]))
        # goal_pos[flat_group_idx, 0] = [i*config['radius'] for i in range(1, rope_particle_num - h + 1)]
        goal_pos[vertical_group_idx, 1] = list(reversed([0.5*i*config['radius'] for i in range(h)]))
        goal_pos[flat_group_idx, 0] = [0.5*i*config['radius'] for i in range(1, rope_particle_num - h + 1)]
        goal_pos[flat_group_idx, 1] = 0.001755
        curr_pos[:,:3] = goal_pos
        pyflex.set_positions(curr_pos.flatten())
        pyflex.set_velocities(np.zeros_like(goal_pos.flatten()))
        pyflex.step()

        center_object()

        # read goal state
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self.key_point_indices, :3]
        goal = keypoint_pos.flatten()

        # visualize the goal scene
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([10, 10, 10])
        self.goal_img = self.render_goal(200, 200)
        return goal

    def _reset(self):
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num, self._num_key_points)

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(rope_particle_num, self._num_key_points), :3]
            picker_radius = self.action_tool.picker_radius
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))
        
        # if hasattr(self, 'action_tool'):
        #     particle_pos = pyflex.get_positions().reshape(-1, 4)
        #     drop_point_pos = particle_pos[self._get_drop_point_idx(rope_particle_num, self._num_key_points), :3]
        #     middle_point = np.mean(drop_point_pos, axis=0)
        #     # self.action_tool.reset(middle_point)  # middle point is not really useful
        #     # picker_radius = self.action_tool.picker_radius
        #     # self.action_tool.update_picker_boundary([-0.3, 0.5, -0.5], [0.5, 2, 0.5])
        #     picker_radius = self.action_tool.picker_radius
        #     self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))

        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _get_drop_point_idx(self,rope_particle_num, num_key_points):
        return self._get_key_point_idx(rope_particle_num, num_key_points)[0]
    
    def _get_flat_pos(self):
        config = self.self.current_config
        rope_particle_num = config['segment'] + 1

        x = np.array([i*config['radius'] for i in range(rope_particle_num)])
        x = x - np.mean(x)

        curr_pos = np.zeros([rope_particle_num, 3], dtype=np.float32)

        curr_pos[:, 0] = x.flatten()
        curr_pos[:, 1] = config['radius']

        # x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        # y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        # x = x - np.mean(x)
        # y = y - np.mean(y)
        # xx, yy = np.meshgrid(x, y)

        # curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        # curr_pos[:, 0] = xx.flatten()
        # curr_pos[:, 2] = yy.flatten()
        # curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_vertical_pos(self, x_low, height_low):
        config = self.current_config
        rope_particle_num = config['segment'] + 1

        x = np.array([i*config['radius'] for i in range(rope_particle_num)])
        x = x - np.mean(x)

        curr_pos = np.zeros([rope_particle_num, 3], dtype=np.float32)

        curr_pos[:,0]

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


