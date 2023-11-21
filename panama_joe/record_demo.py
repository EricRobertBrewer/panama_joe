import argparse
import pickle

import gymnasium as gym
import pygame
from gymnasium.utils import play
from shimmy.atari_env import AtariEnv

from panama_joe.wrappers import AtariDemo

KEY_TO_MEANING = {
    tuple(): 'NOOP',
    (pygame.K_SPACE,): 'FIRE',
    (pygame.K_UP,): 'UP',
    (pygame.K_RIGHT,): 'RIGHT',
    (pygame.K_LEFT,): 'LEFT',
    (pygame.K_DOWN,): 'DOWN',
    (pygame.K_UP, pygame.K_RIGHT): 'UPRIGHT',
    (pygame.K_UP, pygame.K_LEFT): 'UPLEFT',
    (pygame.K_DOWN, pygame.K_RIGHT): 'DOWNRIGHT',
    (pygame.K_DOWN, pygame.K_LEFT): 'DOWNLEFT',
    (pygame.K_UP, pygame.K_SPACE): 'UPFIRE',
    (pygame.K_RIGHT, pygame.K_SPACE): 'RIGHTFIRE',
    (pygame.K_LEFT, pygame.K_SPACE): 'LEFTFIRE',
    (pygame.K_DOWN, pygame.K_SPACE): 'DOWNFIRE',
    (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 'UPRIGHTFIRE',
    (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 'UPLEFTFIRE',
    (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 'DOWNRIGHTFIRE',
    (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 'DOWNLEFTFIRE',
    (pygame.K_s,): 'SAVE'
}


def main():
    parser = argparse.ArgumentParser(
        description='Record demonstrations.'
    )
    parser.add_argument('--env_id',
                        default='ALE/MontezumaRevenge-v5',
                        type=str,
                        help='Environment ID.')
    parser.add_argument('--fps',
                        default=30,
                        type=int,
                        help='Frames per second.')
    parser.add_argument('--zoom',
                        default=3.0,
                        type=float,
                        help='Ratio to render frame from the default screen size.')
    parser.add_argument('--obs_type',
                        default='ram',
                        choices=('ram', 'rgb'),
                        help='Observation type.')
    parser.add_argument('--frameskip',
                        default=1,
                        type=int,
                        help='Frames to skip.')
    parser.add_argument('--repeat_action_probability',
                        default=0.0,
                        type=float,
                        help='Stickiness.')

    args = parser.parse_args()
    kwargs = dict(vars(args))
    record_demo(**kwargs)


def record_demo(env_id, fps=30, zoom=3.0, obs_type='ram', frameskip=1, repeat_action_probability=0.0):
    pygame.init()
    pygame.display.set_caption(f'Recording demonstration for {env_id}')

    env = AtariDemo(gym.make(env_id, obs_type=obs_type, frameskip=frameskip, repeat_action_probability=repeat_action_probability, render_mode='rgb_array'))

    def callback(obs_t_, obs_tp1_, action_, rew_, terminated_, truncated_, info_):
        if isinstance(obs_t_, tuple):  # Very first callback comes with extra `info` object.
            obs_t_ = obs_t_[0]
        print(f'obs_t.shape={obs_t_.shape}; obs_tp1.shape={obs_tp1_.shape}; action={action_}; rew={rew_}; terminated={terminated_}; truncated={truncated_}; info={info_}')

    atari_env = env.unwrapped
    assert isinstance(atari_env, AtariEnv)
    meanings = atari_env.get_action_meanings() + ['SAVE']
    meaning_to_index = {meaning: i for i, meaning in enumerate(meanings)}
    key_to_index = {key: meaning_to_index[meaning] for key, meaning in KEY_TO_MEANING.items()}
    play.play(env, fps=fps, zoom=zoom, callback=callback, keys_to_action=key_to_index, noop=meaning_to_index['NOOP'])


if __name__ == '__main__':
    main()
