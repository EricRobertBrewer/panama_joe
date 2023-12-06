import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3.ppo import PPO

from panama_joe.utils import folders, montezuma


def main():
    parser = argparse.ArgumentParser(
        description='Give TAMER-style feedback to a learning algorithm.'
    )
    parser.add_argument('model_name',
                        choices=('ppo',),
                        help='Algorithm name.')
    parser.add_argument('--num_episodes',
                        '-e',
                        default=1,
                        type=int,
                        help='Number of episodes to evaluate.')
    parser.add_argument('--seed',
                        type=int,
                        help='Seed to random number generator.')
    args = parser.parse_args()
    kwargs = dict(vars(args))
    binary_feedback(**kwargs)


def binary_feedback(model_name, num_episodes=1, seed=None):
    # env = montezuma.make_env(render_mode='human')
    env = montezuma.make_env(obs_type='rgb', render_mode='human')
    if model_name == 'ppo':
        # model = PPO('MlpPolicy', env, seed=seed)
        model = PPO('CnnPolicy', env, seed=seed)
    else:
        raise ValueError(f'Unknown algorithm name: {model_name}')

    # optimizer = torch.optim.Adam([model.get_parameters()])
    optimizer = model.policy.optimizer
    learn_with_input(env, model, optimizer, num_episodes, seed=seed)
    binary_dir = folders.model_dir('binary')
    model_dir = os.path.join(binary_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    stamp = int(time.time())
    path = os.path.join(model_dir, f'{stamp:d}')
    model.save(path)


def learn_with_input(env, model, optimizer, num_episodes, seed=None):
    env.reset(seed=seed)
    episode = 0
    while episode < num_episodes:
        total_reward = 0
        obs, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            # obs_tensor = torch.tensor(np.transpose(obs, axes=(2, 0, 1)))
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            feedback = input('ENTER: no reward; 1: reward! - ')
            if feedback == '1':
                # reward += 1
                optimizer.zero_grad()
                obs_tensor, vec_env = model.policy.obs_to_tensor(obs)
                logits = model.policy.predict_values(obs_tensor)
                print(logits)
                loss = 1 - logits[0][0]
                loss.backward()
                optimizer.step()

            total_reward += reward
        print(f'Total reward: {total_reward:d}')
        episode += 1


if __name__ == '__main__':
    main()
