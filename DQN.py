from stable_baselines3 import DQN
from task1 import Task1
from task2 import Task2
import time

env = Task1(grid_size=8, max_steps=100, render_mode="ansi")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

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