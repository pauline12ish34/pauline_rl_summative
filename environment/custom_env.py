#!/usr/bin/env python3
"""
Custom Gymnasium environment for autonomous warehouse robot
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class WarehouseEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment parameters
        self.grid_size = 8
        self.max_steps = 200
        self.max_inventory = 5
        
        # Action space: 0=North, 1=South, 2=East, 3=West, 4=Pick, 5=Drop
        self.action_space = spaces.Discrete(6)
        
        # Observation space: multi-plane encoding (robot + items + targets + scalars)
        grid_size = self.grid_size * self.grid_size
        obs_size = grid_size * 3 + 4
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Robot starts at bottom-left
        self.robot_pos = [1, self.grid_size-2]
        self.carrying = 0
        self.steps = 0
        
        # Generate items (3 items)
        self.items = [[2, 2], [5, 1], [6, 5]]
        
        # Target zones (2 targets)
        self.targets = [[7, 0], [0, 0]]
        
        # Dynamic obstacles
        self.obstacles = [[3, 3], [4, 5], [1, 4]]
        self.obstacle_directions = [[1, 0], [0, 1], [-1, 0]]
        
        self.delivered_items = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.steps += 1
        reward = -0.05  # Small time penalty
        terminated = False
        
        action = int(action)
        
        # Move obstacles
        self._move_obstacles()
        
        # Execute action
        if action < 4:  # Movement
            reward += self._move(action)
        elif action == 4:  # Pick
            reward += self._pick_item()
        elif action == 5:  # Drop
            reward += self._drop_item()
        
        # Reward shaping
        items_collected = 3 - len(self.items)
        if items_collected == 3 and self.carrying == 0:
            reward += 1000  # Mission complete bonus
            terminated = True
        elif items_collected >= 2 and len(self.items) == 1:
            reward += 5.0  # Final item bonus
        
        # Check termination
        if self.steps >= self.max_steps:
            terminated = True
            if len(self.items) > 0 or self.carrying > 0:
                reward -= 20
        
        return self._get_observation(), reward, terminated, False, {}
    
    def _move_obstacles(self):
        for i, (obs, direction) in enumerate(zip(self.obstacles, self.obstacle_directions)):
            new_x = obs[0] + direction[0]
            new_y = obs[1] + direction[1]
            
            if new_x <= 0 or new_x >= self.grid_size-1:
                self.obstacle_directions[i][0] *= -1
            if new_y <= 0 or new_y >= self.grid_size-1:
                self.obstacle_directions[i][1] *= -1
            
            self.obstacles[i][0] = max(1, min(self.grid_size-2, new_x))
            self.obstacles[i][1] = max(1, min(self.grid_size-2, new_y))
    
    def _move(self, action):
        old_pos = self.robot_pos.copy()
        moves = {0: [0, -1], 1: [0, 1], 2: [1, 0], 3: [-1, 0]}
        dx, dy = moves[action]
        
        new_x = max(0, min(self.grid_size - 1, self.robot_pos[0] + dx))
        new_y = max(0, min(self.grid_size - 1, self.robot_pos[1] + dy))
        
        if [new_x, new_y] in self.obstacles:
            return -10  # Collision penalty
        
        self.robot_pos = [new_x, new_y]
        
        # Reward for moving closer to objectives
        if self.carrying == 0 and self.items:
            nearest_item = min(self.items, key=lambda x: abs(x[0] - new_x) + abs(x[1] - new_y))
            old_dist = abs(nearest_item[0] - old_pos[0]) + abs(nearest_item[1] - old_pos[1])
            new_dist = abs(nearest_item[0] - new_x) + abs(nearest_item[1] - new_y)
            if new_dist < old_dist:
                return 1.0
        elif self.carrying > 0:
            nearest_target = min(self.targets, key=lambda x: abs(x[0] - new_x) + abs(x[1] - new_y))
            old_dist = abs(nearest_target[0] - old_pos[0]) + abs(nearest_target[1] - old_pos[1])
            new_dist = abs(nearest_target[0] - new_x) + abs(nearest_target[1] - new_y)
            if new_dist < old_dist:
                return 1.0
        
        return 0
    
    def _pick_item(self):
        if self.carrying >= self.max_inventory:
            return -2
        
        for i, item in enumerate(self.items):
            if item == self.robot_pos:
                self.items.pop(i)
                self.carrying += 1
                
                # Progressive pickup rewards
                items_remaining = len(self.items)
                if items_remaining == 0:  # Last item
                    return 80
                else:
                    return 50
        
        return -2
    
    def _drop_item(self):
        if self.carrying == 0:
            return -2
        
        for target in self.targets:
            if target == self.robot_pos:
                self.carrying -= 1
                self.delivered_items += 1
                
                # Progressive delivery rewards
                if len(self.items) == 0 and self.carrying == 0:
                    return 300  # Final delivery bonus
                elif self.delivered_items == 2:
                    return 120  # Second delivery
                else:
                    return 100  # First delivery
        
        return -5
    
    def _get_observation(self):
        grid_size = self.grid_size * self.grid_size
        obs = np.zeros(grid_size * 3 + 4, dtype=np.float32)
        
        # Robot position (one-hot)
        robot_idx = self.robot_pos[1] * self.grid_size + self.robot_pos[0]
        obs[robot_idx] = 1.0
        
        # Items (one-hot for each item)
        for item in self.items:
            item_idx = grid_size + item[1] * self.grid_size + item[0]
            obs[item_idx] = 1.0
        
        # Targets (one-hot for each target)
        for target in self.targets:
            target_idx = 2 * grid_size + target[1] * self.grid_size + target[0]
            obs[target_idx] = 1.0
        
        # Scalar features
        obs[-4] = self.carrying / self.max_inventory
        obs[-3] = (self.max_steps - self.steps) / self.max_steps
        obs[-2] = len(self.items) / 5
        obs[-1] = self.delivered_items / 5
        
        return obs
    
    def render(self):
        from .rendering import render_warehouse
        if self.render_mode == "human":
            return render_warehouse(self, mode="human")
        elif self.render_mode == "rgb_array":
            return render_warehouse(self, mode="rgb_array")
        return None
    
    def close(self):
        pass