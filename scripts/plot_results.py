"""
Plotting script for RL experiment summaries: avg_reward vs. key hyperparameters
Uses *_experiments.csv files in results/ directory.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

algos = [
    ("DQN", "results/dqn_experiments.csv", "learning_rate"),
    ("REINFORCE", "results/reinforce_experiments.csv", "learning_rate"),
    ("PPO", "results/ppo_experiments.csv", "learning_rate")
]


# Ensure the plots directory exists
plots_dir = "results/plots"
os.makedirs(plots_dir, exist_ok=True)

for algo, path, param in algos:
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path)
    plt.figure(figsize=(8,5))
    if param in df.columns and "avg_reward" in df.columns:
        plot_path = os.path.join(plots_dir, f"{algo.lower()}_avg_reward_vs_{param}.png")
        plt.scatter(df[param], df["avg_reward"], label=f"{algo} avg_reward")
        plt.xlabel(param)
        plt.ylabel("Average Reward")
        plt.title(f"{algo}: Average Reward vs. {param}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")
    else:
        print(f"Required columns not found in {path}")


# --- Additional Visualizations ---
# 1. Reward curves for all methods in a single plot (if possible)
import matplotlib.pyplot as plt

reward_curves = {}
for algo, path, param in algos:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if "avg_reward" in df.columns:
        reward_curves[algo] = df["avg_reward"].values

if reward_curves:
    plt.figure(figsize=(10,6))
    for algo, rewards in reward_curves.items():
        plt.plot(rewards, label=algo)
    plt.xlabel("Experiment #")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Curves (All Methods)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all_methods_reward_curves.png"))
    plt.close()
    print(f"Saved: {os.path.join(plots_dir, 'all_methods_reward_curves.png')}")

# 2. DQN objective curves (if loss column exists)
dqn_path = "results/dqn_results.csv"
if os.path.exists(dqn_path):
    dqn_df = pd.read_csv(dqn_path)
    if "loss" in dqn_df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(dqn_df["loss"], label="DQN Loss")
        plt.xlabel("Step/Episode")
        plt.ylabel("Loss")
        plt.title("DQN Objective Curve (Loss)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "dqn_objective_curve.png"))
        plt.close()
        print(f"Saved: {os.path.join(plots_dir, 'dqn_objective_curve.png')}")

    # DQN Entropy Curve (if entropy column exists)
    if "entropy" in dqn_df.columns and not dqn_df["entropy"].isnull().all():
        plt.figure(figsize=(8,5))
        plt.plot(dqn_df["entropy"], label="DQN Entropy", color="green")
        plt.xlabel("Step/Episode")
        plt.ylabel("Entropy")
        plt.title("DQN Entropy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "dqn_entropy_curve.png"))
        plt.close()
        print(f"Saved: {os.path.join(plots_dir, 'dqn_entropy_curve.png')}")

# 3. Policy Gradient (PG) entropy curves (if entropy column exists)
for algo in ["REINFORCE", "PPO"]:
    path = f"results/{algo.lower()}_results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Entropy curve
        if "entropy" in df.columns and not df["entropy"].isnull().all():
            plt.figure(figsize=(8,5))
            plt.plot(df["entropy"], label=f"{algo} Entropy")
            plt.xlabel("Step/Episode")
            plt.ylabel("Entropy")
            plt.title(f"{algo} Entropy Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{algo.lower()}_entropy_curve.png"))
            plt.close()
            print(f"Saved: {os.path.join(plots_dir, f'{algo.lower()}_entropy_curve.png')}")
        # PPO Loss curve
        if algo == "PPO" and "loss" in df.columns and not df["loss"].isnull().all():
            plt.figure(figsize=(8,5))
            plt.plot(df["loss"], label="PPO Loss", color="red")
            plt.xlabel("Step/Episode")
            plt.ylabel("Loss")
            plt.title("PPO Loss Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "ppo_loss_curve.png"))
            plt.close()
            print(f"Saved: {os.path.join(plots_dir, 'ppo_loss_curve.png')}")

# 4. Convergence plots (reward over time for each method, if available)
for algo in ["DQN", "REINFORCE", "PPO"]:
    path = f"results/{algo.lower()}_results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "reward" in df.columns:
            plt.figure(figsize=(8,5))
            plt.plot(df["reward"], label=f"{algo} Reward")
            plt.xlabel("Step/Episode")
            plt.ylabel("Reward")
            plt.title(f"{algo} Convergence Plot (Reward)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{algo.lower()}_convergence_plot.png"))
            plt.close()
            print(f"Saved: {os.path.join(plots_dir, f'{algo.lower()}_convergence_plot.png')}")

# 5. Generalization tests (if you have test results, plot them here)
# (Placeholder: add your test results plotting here if available)

print("All required visualizations generated.")

# Compare best avg_reward for each model in a bar chart
import numpy as np

best_rewards = []
labels = []
for algo, path, _ in algos:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if "avg_reward" in df.columns:
        best = df["avg_reward"].max()
        best_rewards.append(best)
        labels.append(algo)

if best_rewards:
    plt.figure(figsize=(7,5))
    plt.bar(labels, best_rewards, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Best Average Reward")
    plt.title("Best Average Reward by Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "best_avg_reward_comparison.png"))
    plt.close()
    print(f"Saved: {os.path.join(plots_dir, 'best_avg_reward_comparison.png')}")
