import numpy as np

class cognitive_map():
    def __init__(self):
        pass
    
    def cognitive_map_task1(agent_pos, goal_pos, map_size = (8,8)):
        map = np.zeros(map_size)
        x, y = agent_pos
        map[x, y] = 1  # Mark current position
        goal_x, goal_y = goal_pos
        map[goal_x, goal_y] = 2  # Mark goal position

        return map
    
    def cognitive_map_random(agent_pos, goal_pos, map_size = (3,3)):
        map = np.zeros(map_size)
        x, y = agent_pos
        goal_x, goal_y = goal_pos
        if x > goal_x:
            target_x = 0
        elif x < goal_x:
            target_x = 2
        else:
            target_x = 1
        if y > goal_y:
            target_y = 0
        elif y < goal_y:
            target_y = 2
        else:
            target_y = 1

        map[target_x, target_y] = 1  # Mark current position
        return map