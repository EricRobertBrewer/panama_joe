import gymnasium as gym

ENV_ID = 'ALE/MontezumaRevenge-v5'


def make_env(obs_type='ram', frameskip=1, repeat_action_probability=0.0, render_mode='rgb_array'):
    """
    The 'ram', 'NoFrameskip', and 'Deterministic' variant of Montezuma's Revenge.
    https://gymnasium.farama.org/environments/atari/montezuma_revenge/

    :param obs_type: Either 'ram' (internal memory, shape=(128,)) or 'rgb' (pixels, shape=(210, 160, 3))
    :param frameskip: Number of frames to skip.
    :param repeat_action_probability: "Stickiness" of actions.
    :param render_mode: 'rgb_array' to visualize.
    :return: Environment.
    """
    return gym.make(ENV_ID,
                    obs_type=obs_type,
                    frameskip=frameskip,
                    repeat_action_probability=repeat_action_probability,
                    render_mode=render_mode)
