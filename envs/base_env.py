from __future__ import annotations
import sys
from typing import Optional, Dict, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BaseGridWorldEnv(gym.Env, ABC):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 10,
    }
    def __init__(
        self,
        render_mode: Optional[str] = None,
        grid_size: int = 8,
        max_steps: int = 200,
        entities: Optional[Dict[int, int]] = None,
        seed: Optional[int] = None,
        cognitive_map: bool = False,
        background: int = 0
    ):
        """
        参数:
        - render_mode: "human" | "rgb_array" | "ansi" | None
        - grid_size: 方格大小，默认为 8
        - max_steps: 每个 episode 的最大步数，防止无限长
        - seed: 随机种子
        """
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.cognitive_map = cognitive_map
        # self.entities = entities or {2: 2, 3: 1}  # 可自行扩展，如 4:1 等
        self._step_count = 0
        self.fill_value = background

        # 观测是一个 grid_size x grid_size 的整数矩阵，0=空，1=agent，>=2 其他事物
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_size, self.grid_size),
            dtype=np.uint8,
        )
        # 动作: 0=上, 1=右, 2=下, 3=左
        self.action_space = spaces.Discrete(4)

        # RNG
        self._rng = np.random.default_rng(seed)

        # 状态
        self.grid: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.target_pos: Tuple[int, int] = (0, 0)
        self.wall_locs = [(grid_size // 2, 0), (1, grid_size // 2), (grid_size // 2 + 1, grid_size - 1)]
        self.trap_locs = [(2, 2), (grid_size - 3, grid_size - 3)]

        # 渲染相关
        self._fig = None
        self._ax = None
        self._im = None
        self._needs_redraw = True

        if self.render_mode not in (None, "human", "rgb_array", "ansi"):
            raise ValueError(f"render_mode 必须为 {self.metadata['render_modes']} 或 None")

    @abstractmethod
    def _spawn(self):
        # 由子类实现的初始化网格的方式
        pass

    def _get_obs(self) -> np.ndarray:
        return self.grid.copy()

    def _get_info(self, action=None) -> Dict[str, Any]:
        if action is not None:
            # 动作: 0=上, 1=右, 2=下, 3=左
            directions = {0: "up", 1: "right", 2: "down", 3: "left"}
            if isinstance(action, np.ndarray):
                action = action.item()
            return {"agent_pos": self.agent_pos, "steps": self._step_count, "direction": directions.get(action, "unknown")}
        return {"agent_pos": self.agent_pos, "steps": self._step_count}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._spawn()
        self._needs_redraw = True

        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            pass  # 无需预渲染

        return self._get_obs(), self._get_info()

    @abstractmethod
    def step(self, action: int):
        pass

    def render(self):
        if self.render_mode == "human":
            self._render_human()
            return None
        elif self.render_mode == "rgb_array":
            return self._grid_to_rgb()
        elif self.render_mode == "ansi":
            return self._grid_to_text()
        else:
            return None

    def _ensure_fig(self):
        if self._fig is not None:
            return
        try:
            import matplotlib
            # Mac 上使用默认 backend 即可；在无显示环境可切换为 Agg
            matplotlib.use(matplotlib.get_backend())
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError(
                "需要 matplotlib 才能进行渲染，请先安装：pip install matplotlib"
            ) from e

        self._plt = plt
        self._plt.ion()
        self._fig, self._ax = self._plt.subplots(figsize=(6, 6))
        self._ax.axis("off")

    def _grid_to_rgb(self) -> np.ndarray:
        h, w = self.grid.shape
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        def paint(mask: np.ndarray, color: Tuple[int, int, int]):
            img[mask] = color
        # 颜色映射
        paint(self.grid == 0, (245, 245, 245))  # empty: 微微灰色
        paint(self.grid == 1, (30, 144, 255))   # agent: dodgerblue
        paint(self.grid == 2, (34, 139, 34))    # goal: green
        paint(self.grid == 3, (255, 0, 0))      # danger: red
        paint(self.grid == 4, (169, 169, 169))  # wall: darkgray
        unknown_mask = (self.grid >= 5)
        if unknown_mask.any():
            paint(unknown_mask, (0, 0, 0))
        return img

    def _render_human(self):
        self._ensure_fig()
        if not self._needs_redraw:
            return
        img = self._grid_to_rgb()

        if self._im is None:
            self._im = self._ax.imshow(img, interpolation="nearest")
        else:
            self._im.set_data(img)

        self._plt.draw()
        # 控制帧率
        pause_time = 1.0 / float(self.metadata.get("render_fps", 10))
        self._plt.pause(pause_time)
        # self._needs_redraw = False

    def _grid_to_text(self) -> str:
        # 用空格分隔，方便阅读
        rows = [" ".join(f"{v:2d}" for v in row) for row in self.grid]
        return "\n".join(rows)

    def close(self):
        if self._fig is not None:
            try:
                self._plt.close(self._fig)
            except Exception:
                pass
            self._fig = None
            self._ax = None
            self._im = None