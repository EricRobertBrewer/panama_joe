import argparse
import pickle
from typing import Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import Env
from gymnasium.core import ActType
from gymnasium.utils import play
from shimmy.atari_env import AtariEnv

from panama_joe import folders, montezuma
from panama_joe.wrappers import AtariDemo

MEANING_SAVE = 'SAVE'
MEANING_TIME_TRAVEL = 'TIMETRAVEL'
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
    (pygame.K_s,): MEANING_SAVE,
    (pygame.K_t,): MEANING_TIME_TRAVEL
}


def main():
    parser = argparse.ArgumentParser(
        description='Record demonstrations.'
    )
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
    parser.add_argument('--demo_file_name',
                        '-d',
                        type=str,
                        help='Load demo from file.')

    args = parser.parse_args()
    kwargs = dict(vars(args))
    record_demo(**kwargs)


def record_demo(fps=30, zoom=3.0, obs_type='ram', frameskip=1, repeat_action_probability=0.0, demo_file_name=None):
    pygame.init()
    pygame.display.set_caption(f'Recording demonstration for {montezuma.ENV_ID}')

    def callback(obs_t_, obs_tp1_, action_, rew_, terminated_, truncated_, info_):
        try:
            print(f'obs_t.shape={obs_t_.shape}; obs_tp1.shape={obs_tp1_.shape}; action={action_}; rew={rew_}; terminated={terminated_}; truncated={truncated_}; info={info_}')
        except AttributeError:
            pass

    env = montezuma.make_env(obs_type=obs_type,
                             frameskip=frameskip,
                             repeat_action_probability=repeat_action_probability,
                             render_mode='rgb_array')  # The "real" environment.
    env = AtariDemo(env, demos_dir=folders.demo_dir('human'))  # Enable saving, time traveling.
    atari_env = env.unwrapped
    assert isinstance(atari_env, AtariEnv)
    meanings = atari_env.get_action_meanings() + [MEANING_SAVE, MEANING_TIME_TRAVEL]
    meaning_to_index = {meaning: i for i, meaning in enumerate(meanings)}
    key_to_index = {key: meaning_to_index[meaning] for key, meaning in KEY_TO_MEANING.items()}
    play_game(env, fps=fps, zoom=zoom, callback=callback, keys_to_action=key_to_index, noop=meaning_to_index['NOOP'], demo_file_name=demo_file_name)


def play_game(
        env: Env,
        transpose: Optional[bool] = True,
        fps: Optional[int] = None,
        zoom: Optional[float] = None,
        callback: Optional[Callable] = None,
        keys_to_action: Optional[Dict[Union[Tuple[Union[str, int]], str], ActType]] = None,
        seed: Optional[int] = None,
        noop: ActType = 0,
        demo_file_name: Optional[str] = None
):
    """
    Copied from `play.play()` except for usage of `demo_file_name` on reset.
    """
    env.reset(seed=seed)

    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert env.spec is not None
            raise MissingKeysToAction(
                f"{env.spec.id} does not have explicit key to action mapping, "
                "please specify one manually"
            )
    assert keys_to_action is not None

    key_code_to_action = {}
    for key_combination, action in keys_to_action.items():
        key_code = tuple(
            sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
        )
        key_code_to_action[key_code] = action

    game = play.PlayableGame(env, key_code_to_action, zoom)

    if fps is None:
        fps = env.metadata.get("render_fps", 30)

    done, obs = True, None
    clock = pygame.time.Clock()

    while game.running:
        if done:
            done = False
            if demo_file_name is not None:
                obs = env.load_from_file(demo_file_name)
            else:
                obs = env.reset(seed=seed)
        else:
            action = key_code_to_action.get(tuple(sorted(game.pressed_keys)), noop)
            prev_obs = obs
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if callback is not None:
                callback(prev_obs, obs, action, rew, terminated, truncated, info)
        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, List):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)
            play.display_arr(
                game.screen, rendered, transpose=transpose, video_size=game.video_size
            )

        # process pygame events
        for event in pygame.event.get():
            game.process_event(event)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


if __name__ == '__main__':
    main()
