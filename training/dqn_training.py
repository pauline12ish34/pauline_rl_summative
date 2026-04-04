#!/usr/bin/env python3
"""
Training script for DQN using Stable-Baselines3
"""
import os
import sys
import random
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from environment.custom_env import WarehouseEnv

def train_dqn_experiment(params, run_id):
    os.makedirs("./models/dqn/", exist_ok=True)
    os.makedirs("./results/dqn_logs/", exist_ok=True)
    env = make_vec_env(lambda: WarehouseEnv(), n_envs=4)
    eval_env = make_vec_env(lambda: WarehouseEnv(), n_envs=1)

    from stable_baselines3.common.callbacks import BaseCallback
    from torch.utils.tensorboard import SummaryWriter
    import torch

    class DQNEntropyCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.writer = None
        def _on_training_start(self):
            log_dir = self.model.logger.dir if hasattr(self.model.logger, 'dir') else './results/dqn_tensorboard/'
            self.writer = SummaryWriter(log_dir)
        def _on_step(self) -> bool:
            # Entropy is not logged by default; estimate from Q-values
            if hasattr(self.model, 'q_net') and hasattr(self.model, 'replay_buffer'):
                # Sample a batch from the replay buffer
                if self.model.replay_buffer.size() > 0:
                    batch = self.model.replay_buffer.sample(64)
                    with torch.no_grad():
                        q_values = self.model.q_net(torch.as_tensor(batch.observations).float().to(self.model.device))
                        probs = torch.softmax(q_values, dim=1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
                        self.writer.add_scalar('custom/entropy', entropy, self.num_timesteps)
            return True
        def _on_training_end(self):
            if self.writer:
                self.writer.close()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=params['learning_rate'],
        buffer_size=params['buffer_size'],
        learning_starts=params['learning_starts'],
        batch_size=params['batch_size'],
        tau=params['tau'],
        gamma=params['gamma'],
        train_freq=params['train_freq'],
        gradient_steps=params['gradient_steps'],
        target_update_interval=params['target_update_interval'],
        exploration_fraction=params['exploration_fraction'],
        exploration_initial_eps=params['exploration_initial_eps'],
        exploration_final_eps=params['exploration_final_eps'],
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        tensorboard_log="./results/dqn_tensorboard/"
    )

    # Per-episode logging (append mode, add entropy column, log experiment index)
    episode_rewards = []
    episode_losses = []
    episode_entropies = []
    obs = env.reset()
    for episode in range(100):
        done = [False] * env.num_envs
        total_reward = 0
        losses = []
        entropies = []
        while not all(done):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += sum(reward)
            # Loss and entropy are not directly available from SB3, so we skip or set to None
        episode_rewards.append(total_reward / env.num_envs)
        episode_losses.append(None)
        episode_entropies.append(None)
    # Append per-episode log for every experiment
    file_exists = os.path.isfile("./results/dqn_results.csv")
    with open("./results/dqn_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["experiment", "episode", "reward", "loss", "entropy"])
        for i, (r, l, e) in enumerate(zip(episode_rewards, episode_losses, episode_entropies)):
            writer.writerow([run_id+1, i+1, r, l, e])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=20000,
        deterministic=True,
        render=False
    )
    model.learn(
        total_timesteps=100000,
        callback=eval_callback
    )
    # Evaluate final mean reward
    rewards = []
    for _ in range(5):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward[0]
            if hasattr(eval_env, 'envs') and hasattr(eval_env.envs[0], 'steps') and eval_env.envs[0].steps >= 200:
                break
        rewards.append(total_reward)
    avg_reward = sum(rewards) / len(rewards)
    return avg_reward

def main():
    print("🎯 Running DQN hyperparameter experiments")
    dqn_param_grid = [
        {
            'learning_rate': lr,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'tau': tau,
            'gamma': gamma,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'target_update_interval': target_update_interval,
            'exploration_fraction': exploration_fraction,
            'exploration_initial_eps': exploration_initial_eps,
            'exploration_final_eps': exploration_final_eps
        }
        for lr in [3e-4, 1e-4]
        for buffer_size in [100000, 200000]
        for learning_starts in [1000, 5000]
        for batch_size in [64, 128]
        for tau in [0.9, 1.0]
        for gamma in [0.98, 0.99]
        for train_freq in [8, 16]
        for gradient_steps in [4, 8]
        for target_update_interval in [500, 1000]
        for exploration_fraction in [0.3, 0.4]
        for exploration_initial_eps in [1.0]
        for exploration_final_eps in [0.01, 0.05]
    ]
    random.shuffle(dqn_param_grid)
    dqn_param_grid = dqn_param_grid[:10]
    with open("./results/dqn_experiments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(dqn_param_grid[0].keys()) + ["avg_reward"])
        for i, params in enumerate(dqn_param_grid):
            print(f"DQN Experiment {i+1}/10: {params}")
            avg_reward = train_dqn_experiment(params, i)
            writer.writerow(list(params.values()) + [avg_reward])
    print("✅ DQN hyperparameter experiments completed!")
    # Always run per-episode logging for default params as a separate run
    print("\n🎯 Running DQN per-episode logging for analysis...")
    default_params = {
        'learning_rate': 1e-3,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'batch_size': 32,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.1,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05
    }
    train_dqn_experiment(default_params, 0)
    print("✅ DQN per-episode logging completed! Results in results/dqn_results.csv\n")

if __name__ == "__main__":
    main()