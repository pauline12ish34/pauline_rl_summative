"""
Random Agent Demo for WarehouseEnv
Visualizes the environment with random actions (no learning).
"""

import time
import numpy as np
import os
import imageio
from environment.custom_env import WarehouseEnv


def run_random_agent_until_delivery(max_steps=200, render=True, save_video=True, video_dir="results/random_agent_demo"):
    os.makedirs(video_dir, exist_ok=True)
    episode = 0
    found = False
    while not found:
        episode += 1
        env = WarehouseEnv(render_mode="rgb_array" if save_video else ("human" if render else None))
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        frames = []
        print(f"\nRandom Agent - Episode {episode}")
        while steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if save_video:
                frame = env.render()  # Should return RGB array
                frames.append(frame)
            elif render:
                env.render()
                time.sleep(0.1)
            if terminated or truncated:
                break
        print(f"  Total Reward: {total_reward:.1f}  Deliveries: {env.delivered_items}/3  Steps: {steps}")
        if env.delivered_items >= 1 and save_video and frames:
            video_path = os.path.join(video_dir, f"random_agent_episode_{episode}_with_delivery.gif")
            imageio.mimsave(video_path, frames, fps=10)
            print(f"  Saved video: {video_path}")
            found = True
        env.close()

if __name__ == "__main__":
    run_random_agent_until_delivery()