import argparse
import os

from stable_baselines3.ppo import PPO

from panama_joe.utils import folders, montezuma


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a saved model.'
    )
    parser.add_argument('model_name',
                        choices=('ppo',),
                        help='Algorithm name.')
    parser.add_argument('file_name',
                        type=str,
                        help='Path to saved model.')
    parser.add_argument('--num_episodes',
                        '-e',
                        default=1,
                        type=int,
                        help='Number of episodes to evaluate.')
    parser.add_argument('--save_video',
                        action='store_true',
                        help='Flag to enable saving of a video.')
    parser.add_argument('--seed',
                        type=int,
                        help='Seed to random number generator.')
    args = parser.parse_args()
    kwargs = dict(vars(args))
    evaluate(**kwargs)


def evaluate(model_name, file_name, num_episodes=1, save_video=False, seed=None):
    env = montezuma.make_env(render_mode='human')
    if model_name == 'ppo':
        path = os.path.join(folders.model_dir(model_name), file_name)
        model = PPO.load(path)
    else:
        raise ValueError(f'Unknown algorithm name: {model_name}')

    env.reset(seed=seed)
    episode = 0
    while episode < num_episodes:
        total_reward = 0
        obs, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(f'Total reward: {total_reward:d}')
        episode += 1


if __name__ == '__main__':
    main()
