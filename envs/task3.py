from envs.base_env import BaseGridWorldEnv
import numpy as np
import time
from typing import Optional, Dict, Tuple, Any
from envs.cognitive_map import cognitive_map

ORIGIN_BACKGROUND = 0
AGENT_VALUE = 1
TARGET_VALUE = 2
TRAP_VALUE = 3
WALL_VALUE = 4
UKNOWN = 5

class Task3(BaseGridWorldEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_distance = None
        self.cur_distance = None
        self.pass_trap = False

    def _spawn(self):
        """
        初始化网格：放置 Agent (1), Target (2), 墙壁 (3) 和 陷阱 (4)
        """
        self.grid.fill(self.fill_value)
        grid_size = self.grid_size
        self.agent_pos = (0, 0)
        self.target_pos = (grid_size - 1, grid_size - 1)
        self.grid[self.agent_pos] = AGENT_VALUE
        self.grid[self.target_pos] = TARGET_VALUE

        # 放置墙壁 (WALL_VALUE = 3)
        for i in range(2, grid_size // 2):
            self.grid[i, grid_size // 2] = WALL_VALUE
            self.grid[grid_size // 2, i] = WALL_VALUE
            
        # 放置陷阱 (TRAP_VALUE = 4)
        trap_locs = [(2, 2), (grid_size - 3, grid_size - 3)]
        for r, c in trap_locs:
            # 确保陷阱不覆盖 Agent 或 Target
            if self.grid[r, c] == self.fill_value:
                self.grid[r, c] = TRAP_VALUE
    
    def _get_obs(self):
        if self.cognitive_map:
            return cognitive_map.cognitive_map_task3(self.agent_pos, self.target_pos, None, map_size=(self.grid_size, self.grid_size))
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
        current_pos = self.agent_pos
        self.pre_distance = np.linalg.norm(np.array(current_pos) - np.array(self.target_pos))

        new_pos_list = list(current_pos)
        move_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = move_deltas[action]
        new_r = current_pos[0] + dr
        new_c = current_pos[1] + dc
        new_pos_list[0] = new_r
        new_pos_list[1] = new_c
        
        # 3. 检查边界和墙壁
        candidate_pos = tuple(new_pos_list)
        # 检查是否越界
        r_valid = 0 <= new_r < self.grid_size
        c_valid = 0 <= new_c < self.grid_size
        # 检查新位置是否是墙壁
        is_wall = False
        if r_valid and c_valid and self.grid[candidate_pos] == WALL_VALUE:
            is_wall = True
        if not r_valid or not c_valid or is_wall:
            final_pos = current_pos
        else:
            final_pos = candidate_pos

        if self.pass_trap == False:
            self.grid[current_pos] = self.fill_value
        else:
            self.grid[current_pos] = TRAP_VALUE
            self.pass_trap = False
        self.agent_pos = final_pos
        target_content = self.grid[self.agent_pos]
        self.grid[self.agent_pos] = AGENT_VALUE # Agent 标记为 1
        self.cur_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.target_pos))

        done = self.agent_pos == self.target_pos or self._step_count >= self.max_steps
        reward = 0.0
        if self.agent_pos == self.target_pos:
            reward = 10.0 # 目标奖励
        elif target_content == TRAP_VALUE:
            self.pass_trap = True
            reward = -1.0 # 陷阱奖励 (高额惩罚)
        elif is_wall:
            reward = -0.05 # 撞墙惩罚 (小额惩罚，鼓励学习避免撞墙)
        else:
            reward = (self.pre_distance - self.cur_distance) * 0.1
        
        info = self._get_info()

        return self.grid.copy(), reward, done, False, info

if __name__ == "__main__":
    env = Task3(render_mode="human", grid_size=8, max_steps=100, seed=42)
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample() # 随机动作
        obs, reward, done, _, info = env.step(action)
        env.render()
        time.sleep(0.2)
    
    env.close()