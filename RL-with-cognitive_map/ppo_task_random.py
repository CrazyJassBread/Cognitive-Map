from stable_baselines3 import PPO
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.task_random import Task_random

import time

env = Task_random(grid_size=8, max_steps=200, render_mode="ansi", cognitive_map=True, random = False)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=40000)

env.close()
env = Task_random(grid_size=8, max_steps=200, render_mode="human", cognitive_map=True)
for _ in range(5):
    obs,_ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        print(reward, info)
        time.sleep(0.1)
        env.render() 