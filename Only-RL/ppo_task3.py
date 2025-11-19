# 给迷宫增加障碍物和陷阱
from stable_baselines3 import PPO
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.task3 import Task3
import time

env = Task3(grid_size=8, max_steps=100, render_mode="ansi", background=0)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./log/task3_PPO/")
model.learn(total_timesteps=40000)
env.close()

env = Task3(grid_size=8, max_steps=100, render_mode="human", background=0)
obs,_ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    print(reward, info)
    time.sleep(0.2)
    env.render()
env.close()

time.sleep(1)

env = Task3(grid_size=8, max_steps=100, render_mode="human", background=5)
obs,_ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    print(reward, info)
    time.sleep(0.2)
    env.render()
env.close()