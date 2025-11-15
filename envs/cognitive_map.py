import numpy as np

class cognitive_map():
    def __init__(self):
        pass
    
    def cognitive_map_task1(agent_pos, goal_pos, action, map_size = (8,8)):
        map = np.zeros(map_size)
        x, y = agent_pos
        map[x, y] = 1  # Mark current position
        goal_x, goal_y = goal_pos
        map[goal_x, goal_y] = 2  # Mark goal position

        return map