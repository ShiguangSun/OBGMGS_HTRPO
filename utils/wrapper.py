import gym
import numpy as np

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class ActionNormalizer(gym.Wrapper):
    def __init__(self, env):
        super(ActionNormalizer, self).__init__(env)
        self.ori_action_space = env.action_space
        assert (self.ori_action_space.low > - np.inf).sum() == self.ori_action_space.low.size
        assert (self.ori_action_space.high < np.inf).sum() == self.ori_action_space.high.size
        self.ori_low = self.ori_action_space.low
        self.ori_high = self.ori_action_space.high
        print("Action Space High: {}, Low: {}".format(self.ori_high, self.ori_low))

    def step(self, a):
        a = (a + 1.) / 2 * (self.ori_high - self.ori_low) + self.ori_low
        a = np.clip(a, a_min=self.ori_low, a_max=self.ori_high)
        return self.env.step(a)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


