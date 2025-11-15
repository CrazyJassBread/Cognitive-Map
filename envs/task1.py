from envs.base_env import BaseGridWorldEnv
import gymnasium as gym
import numpy as np
import time

class Task1(BaseGridWorldEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_distance = None
        self.cur_distance = None

    def _spawn(self):
        self.grid.fill(0)
        self.agent_pos = (0,0)
        self.target_pos = (self.grid_size - 1, self.grid_size - 1)
        self.grid[self.agent_pos] = 1  # 代理人用值1表示
        self.grid[self.target_pos] = 2  # 目标用值2表示
    
    def step(self, action):
        self._step_count += 1
        self.pre_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target_pos))

        new_pos = list(self.agent_pos)
        if action == 0 and self.agent_pos[0] > 0:  # 上
            new_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # 右
            new_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.grid_size - 1:  # 下
            new_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:  # 左
            new_pos[1] -= 1

        # 更新位置
        self.grid[self.agent_pos] = 0
        self.agent_pos = tuple(new_pos)
        self.grid[self.agent_pos] = 1
        self.cur_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target_pos))

        # 检查是否到达目标
        done = self.agent_pos == self.target_pos or self._step_count >= self.max_steps
        reward = 10.0 if self.agent_pos == self.target_pos else (self.pre_distance - self.cur_distance)*0.1
        info = self._get_info()

        return self.grid.copy(), reward, done, False, info
    
if __name__ == "__main__":
    env = Task1(render_mode="human", grid_size=8, max_steps=50, seed=42)
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, _, info = env.step(action)
        env.render()
        time.sleep(0.2)
    
    env.close()