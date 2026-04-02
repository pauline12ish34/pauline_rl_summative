#!/usr/bin/env python3
"""
Entry point for running the best performing model
"""
import os
import sys
import pygame
import time
from stable_baselines3 import PPO
from environment.custom_env import WarehouseEnv

def run_best_model():
    """Run the best performing model (PPO)"""
    print("Vaccine Cold-Chain Storage Robot - Rwanda")
    print("=" * 60)
    
    # Load the best model (PPO)
    try:
        model = PPO.load("./models/pg/ppo_final")
        print(" Loaded PPO model (best performer)")
    except FileNotFoundError:
        print(" Best model not found. Please train models first:")
        print("   python3 training/pg_training.py")
        return
    
    # Create environment with rendering
    env = WarehouseEnv(render_mode="human")
    print("\nStarting demonstration...")
    print("Mission: Collect 3 vaccine boxes and deliver to health facilities")
    print("Expected: 3/3 deliveries, ~1781 reward, ~30 steps")

    # Run demonstration episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        print("-" * 40)

        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        print(f"Initial state:")
        print(f"  Robot: {env.robot_pos}")
        print(f"  Vaccine Boxes: {env.items}")
        print(f"  Health Facilities: {env.targets}")

        while steps < 200:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            # Render environment
            env.render()

            # Handle pygame events
            if hasattr(env, 'window') and env.window is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return

            # Control speed for better visualization
            time.sleep(0.1)

            if terminated or truncated:
                break

        # Episode results
        print(f"\nEpisode {episode + 1} Results:")
        print(f"  Total Reward: {total_reward:.1f}")
        print(f"  Deliveries to Health Facilities: {env.delivered_items}/3")
        print(f"  Steps: {steps}")
        print(f"  Vaccine Boxes Remaining: {len(env.items)}")

        if env.delivered_items >= 3:
            print("  MISSION COMPLETE! All vaccines delivered to health facilities!")
        elif env.delivered_items >= 2:
            print("  Good performance! 2 vaccine deliveries completed.")
        else:
            print("  Needs improvement.")

        if episode < 2:
            input("\nPress Enter for next episode...")

    env.close()
    print("\nDemonstration complete!")

def compare_models():
    """Compare all trained models"""
    print("\nModel Performance Comparison")
    print("=" * 60)

    models = {
        "DQN (Value-Based)": "./models/dqn/final_model",
        "PPO (Policy Gradient)": "./models/pg/ppo_final",
    }

    results = {}

    for name, path in models.items():
        if not os.path.exists(path + ".zip"):
            print(f"{name}: Model not found")
            continue

        try:
            if "DQN" in name:
                from stable_baselines3 import DQN
                model = DQN.load(path)
            elif "PPO" in name:
                model = PPO.load(path)

            # Test model
            env = WarehouseEnv()
            total_reward = 0
            total_deliveries = 0
            episodes = 5

            for episode in range(episodes):
                obs, _ = env.reset()
                episode_reward = 0

                for step in range(200):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        break

                total_reward += episode_reward
                total_deliveries += env.delivered_items

            avg_reward = total_reward / episodes
            avg_deliveries = total_deliveries / episodes

            results[name] = {
                'reward': avg_reward,
                'deliveries': avg_deliveries,
                'success_rate': (total_deliveries >= episodes * 2.5)
            }

            print(f"{name}:")
            print(f"   Average Reward: {avg_reward:.1f}")
            print(f"   Average Deliveries: {avg_deliveries:.1f}")
            print(f"   Success Rate: {'High' if results[name]['success_rate'] else 'Low'}")

        except Exception as e:
            print(f"{name}: Error - {e}")

    # Find best model
    if results:
        best = max(results.items(), key=lambda x: x[1]['reward'])
        print(f"\nBest Model: {best[0]}")
        print(f"   Reward: {best[1]['reward']:.1f}")
        print(f"   Deliveries: {best[1]['deliveries']:.1f}")

def main():
    """Main entry point"""
    print("Vaccine Cold-Chain Storage Robot - Rwanda Health Facilities")
    print("=" * 80)
    print("1. Run Best Model Demo")
    print("2. Compare All Models")
    print("3. Exit")

    while True:
        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            run_best_model()
        elif choice == "2":
            compare_models()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()