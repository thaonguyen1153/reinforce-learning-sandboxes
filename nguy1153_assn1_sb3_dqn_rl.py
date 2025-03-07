import gymnasium as gym
import blocksworld_env
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN


env = gym.make("blocksworld_env/BlocksWorld-v1", render_mode="human")

model = DQN("MlpPolicy", env, verbose=1) # MlpPolicy: because the env is using Discrete values for obs and action

model.learn(total_timesteps=10, log_interval=4)#10000
#save a model to a file
model.save("dqn_blocks")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_blocks")

obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)
    #print(f"action:{action} and state: {_states}")
    obs, reward, terminated, truncated, info = env.step(int(action))
    print(f"action:{action} and reward: {reward} and obs: {obs}")
    if terminated or truncated:
        obs, info = env.reset()