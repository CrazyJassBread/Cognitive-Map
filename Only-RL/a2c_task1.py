from stable_baselines3 import A2C
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.task1 import Task1
from envs.task2 import Task2
import time

env = Task1(grid_size=8, max_steps=100, render_mode="ansi")
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./log/task1_A2C/")
model.learn(total_timesteps=40000)
env.close()

env = Task2(grid_size=8, max_steps=100, render_mode="human")
obs,_ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    print(reward, info)
    time.sleep(0.2)
    env.render() 