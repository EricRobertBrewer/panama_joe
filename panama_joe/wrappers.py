import os
import pickle
import time

import gymnasium as gym
from gymnasium import spaces
from shimmy.atari_env import AtariEnv


class AtariDemo(gym.Wrapper):
    """
        Records actions taken, creates checkpoints, allows time travel, restoring and saving of states
    """

    def __init__(self, env, disable_time_travel=False):
        super(AtariDemo, self).__init__(env)
        # self.action_space = spaces.Discrete(len(env.unwrapped._action_set)+1) # add "time travel" action
        self.action_space = spaces.Discrete(len(env.unwrapped.get_action_meanings())+1)  # add "save" action
        self.save_every_k = 100
        self.steps_in_the_past = 0
        self.max_time_travel_steps = 10000
        self.disable_time_travel = disable_time_travel

        self.actions = None
        self.lives = None
        self.checkpoints = None
        self.checkpoint_action_nr = None
        self.obs = None
        self.rewards = None
        self.terminated = None
        self.truncated = None
        self.info = None

    def step(self, action):
        if action == len(self.env.unwrapped.get_action_meanings()) + 1:  # 't': time travel
            if self.disable_time_travel:
                obs, reward, terminated, truncated, info = self.env.step(0)
            else:
                obs, reward, terminated, truncated, info = self.time_travel()

        else:
            if self.steps_in_the_past > 0:
                self.restore_past_state()

            if (len(self.terminated) > 0 and self.terminated[-1]) or (len(self.truncated) > 0 and self.truncated[-1]):
                obs = self.obs[-1]
                reward = 0
                terminated = self.terminated[-1]
                truncated = self.truncated[-1]
                info = None

            else:
                atari_env = self.env.unwrapped
                assert isinstance(atari_env, AtariEnv)
                self.lives.append(atari_env.ale.lives())

                if action == len(atari_env.get_action_meanings()):  # 's': save
                    obs, reward, terminated, truncated, info = self.env.step(0)
                    stamp = int(time.time())
                    self.save_to_file(str(stamp) + '.demo')
                else:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self.actions.append(action)
                    self.obs.append(obs)
                    self.rewards.append(reward)
                    self.terminated.append(terminated)
                    self.truncated.append(truncated)
                    self.info.append(info)

            # periodic checkpoint saving
            if not terminated and not truncated:
                if (len(self.checkpoint_action_nr) > 0 and len(self.actions) >= self.checkpoint_action_nr[-1] + self.save_every_k) \
                        or (len(self.checkpoint_action_nr) == 0 and len(self.actions) >= self.save_every_k):
                    self.save_checkpoint()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset(seed=seed)
        self.actions = []
        self.lives = []
        self.checkpoints = []
        self.checkpoint_action_nr = []
        self.obs = [obs]
        self.rewards = []
        self.terminated = [False]
        self.truncated = [False]
        self.info = [None]
        # self.steps_in_the_past = 0
        return obs

    def time_travel(self):
        if len(self.obs) > 1:
            reward = self.rewards.pop()
            self.obs.pop()
            self.terminated.pop()
            self.truncated.pop()
            self.info.pop()
            self.lives.pop()
            obs = self.obs[-1]
            terminated = self.terminated[-1]
            truncated = self.truncated[-1]
            info = self.info[-1]
            self.steps_in_the_past += 1

        else: # reached time travel limit
            reward = 0
            obs = self.obs[0]
            terminated = self.terminated[0]
            truncated = self.truncated[0]
            info = self.info[0]

        # rewards are differences in subsequent state values, and so should get reversed sign when going backward in time
        reward = -reward

        return obs, reward, terminated, truncated, info

    def save_to_file(self, file_name):
        dat = {'actions': self.actions, 'checkpoints': self.checkpoints, 'checkpoint_action_nr': self.checkpoint_action_nr,
               'obs': self.obs,
               'rewards': self.rewards, 'lives': self.lives}
        with open(file_name, "wb") as f:
            pickle.dump(dat, f)

    def load_from_file(self, file_name):
        obs = self.reset()
        with open(file_name, "rb") as f:
            dat = pickle.load(f)
        self.actions = dat['actions']
        self.checkpoints = dat['checkpoints']
        self.checkpoint_action_nr = dat['checkpoint_action_nr']
        self.obs = dat['obs']
        self.rewards = dat['rewards']
        self.lives = dat['lives']
        self.load_state_and_walk_forward()
        return obs

    def save_checkpoint(self):
        chk_pnt = self.env.unwrapped.clone_state()
        self.checkpoints.append(chk_pnt)
        self.checkpoint_action_nr.append(len(self.actions))

    def restore_past_state(self):
        self.actions = self.actions[:-self.steps_in_the_past]
        while len(self.checkpoints) > 0 and self.checkpoint_action_nr[-1] > len(self.actions):
            self.checkpoints.pop()
            self.checkpoint_action_nr.pop()
        self.load_state_and_walk_forward()
        self.steps_in_the_past = 0

    def load_state_and_walk_forward(self):
        if len(self.checkpoints) == 0:
            self.env.reset()
            time_step = 0
        else:
            self.env.unwrapped.restore_state(self.checkpoints[-1])
            time_step = self.checkpoint_action_nr[-1]

        for a in self.actions[time_step:]:
            action = self.env.unwrapped._action_set[a]
            self.env.unwrapped.ale.act(action)
