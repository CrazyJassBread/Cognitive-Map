import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.task1 import Task1
from envs.task2 import Task2
from envs.task3 import Task3
import time
from pynput import keyboard
def get_action_from_key():
    action_map = {
        'w': 0,    # 上
        'd': 1, # 右
        's': 2,  # 下
        'a': 3,   # 左
    }
    action = None
    def on_press(key):
        nonlocal action
        try:
            # 检查是否是方向键或 w/a/s/d 键
            if key in action_map:
                action = action_map[key]
                return False  # 停止监听
            elif hasattr(key, 'char') and key.char in action_map:
                action = action_map[key.char]
                return False  # 停止监听
        except Exception as e:
            pass  # 忽略所有异常
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    return action

if __name__ == "__main__":
    env = Task3(render_mode="human", grid_size=8, max_steps=100, seed=42,background = 5)
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        action = get_action_from_key()
        if action is None:
            continue
        steps += 1
        obs, reward, done, _, info = env.step(action)
        env.render()
    print(f"Episode finished in {steps} steps.")
    env.close()