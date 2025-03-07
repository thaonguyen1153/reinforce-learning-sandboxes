import gymnasium as gym
import blocksworld_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def train_and_run():
    # Parallel environments
    env = make_vec_env("blocksworld_env/BlocksWorld-v1", n_envs=4, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", env, device="cpu", verbose=1)
    model.learn(total_timesteps=25_000)
    model.save("ppo_blocks")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo_blocks")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f"action:{action} and rewards: {rewards}")
        #env.render("human")

if __name__ == "__main__":
    train_and_run()