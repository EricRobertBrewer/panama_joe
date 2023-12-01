import argparse
import os
import time

from stable_baselines3.ppo import PPO

from panama_joe.utils import folders, montezuma


def main():
    parser = argparse.ArgumentParser(
        description='Train a baseline with which to compare our methods.'
    )
    parser.add_argument('model_name',
                        choices=('ppo',),
                        help='Algorithm name.')
    parser.add_argument('--total_timesteps',
                        '-t',
                        default=2048,
                        type=int,
                        help='Number of steps to run in the environment.')
    parser.add_argument('--seed',
                        type=int,
                        help='Seed to random number generator.')
    args = parser.parse_args()
    kwargs = dict(vars(args))
    train_baseline(**kwargs)


def train_baseline(model_name, total_timesteps=2048, seed=None):
    env = montezuma.make_env()
    if model_name == 'ppo':
        model = PPO('MlpPolicy', env, seed=seed)
    else:
        raise ValueError(f'Unknown algorithm name: {model_name}')

    model.learn(total_timesteps=total_timesteps, log_interval=16)
    model_dir = folders.model_dir(model_name)
    stamp = int(time.time())
    path = os.path.join(model_dir, f'{stamp:d}')
    model.save(path)


if __name__ == '__main__':
    main()
