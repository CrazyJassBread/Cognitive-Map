from envs.base_env import BaseGridWorldEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from typing import Optional, Dict, Tuple, Any
from envs.cognitive_map import cognitive_map

class Task_random(BaseGridWorldEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_distance = None
        self.cur_distance = None
        
        if getattr(self, "cognitive_map", False):
            obs_shape = (3, 3)
        else:
            obs_shape = (self.grid_size, self.grid_size)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def _spawn(self):
        self.grid.fill(0)
        rng = getattr(self, "_rng", None) or np.random.default_rng()

        total_cells = self.grid_size * self.grid_size
        # 从所有格子中不放回抽样两个不同的索引
        a_idx, t_idx = rng.choice(total_cells, size=2, replace=False)

        def idx_to_pos(i):
            return (i // self.grid_size, i % self.grid_size)

        self.agent_pos = idx_to_pos(int(a_idx))
        self.target_pos = idx_to_pos(int(t_idx))

        self.grid[self.agent_pos] = 1  # 代理人用值1表示
        self.grid[self.target_pos] = 2  # 目标用值2表示


    def _get_obs(self):
        if self.cognitive_map:
            return cognitive_map.cognitive_map_random(self.agent_pos, self.target_pos)
        else:
            return self.grid.copy()
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._spawn()
        self._needs_redraw = True

        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            pass
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self._step_count += 1
        reward = 0.0
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

        if self.agent_pos == self.target_pos:
            reward += 30.0 
        else:
            reward += (self.pre_distance - self.cur_distance) * 1

        if self._step_count >= 20:
            reward -= 0.1  # 每步惩罚

        self.pre_distance = self.cur_distance
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, done, False, info
    
if __name__ == "__main__":
    env = Task_random(render_mode="human", grid_size=8, max_steps=50, seed=42)
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, _, info = env.step(action)
        env.render()
        time.sleep(0.2)
    
    env.close()