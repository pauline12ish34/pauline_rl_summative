#!/usr/bin/env python3
"""
Training script for Policy Gradient methods (PPO, A2C, REINFORCE) using Stable-Baselines3
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from environment.custom_env import WarehouseEnv

class REINFORCEPolicy(nn.Module):
    """Custom REINFORCE policy network"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(REINFORCEPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

def train_ppo_experiment(params, run_id):
    os.makedirs("./models/pg/", exist_ok=True)
    os.makedirs("./results/ppo_logs/", exist_ok=True)
    env = make_vec_env(lambda: WarehouseEnv(), n_envs=4)
    eval_env = make_vec_env(lambda: WarehouseEnv(), n_envs=1)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=params['learning_rate'],
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        gamma=params['gamma'],
        gae_lambda=params['gae_lambda'],
        clip_range=params['clip_range'],
        ent_coef=params['ent_coef'],
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        tensorboard_log=None
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    model.learn(
        total_timesteps=200000,
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
    avg_reward = np.mean(rewards)
    return avg_reward


def train_reinforce_experiment(params, run_id):
    os.makedirs("./models/pg/", exist_ok=True)
    env = WarehouseEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = REINFORCEPolicy(state_size, action_size, hidden_size=256)
    optimizer = optim.Adam(policy.parameters(), lr=params['learning_rate'])
    episodes = params['episodes']
    gamma = params['gamma']
    all_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        for step in range(200):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            if terminated or truncated:
                break
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # Policy gradient update
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        all_rewards.append(sum(rewards))
    avg_reward = np.mean(all_rewards[-10:])
    return avg_reward

def main():
    print("🎯 Running PPO and REINFORCE hyperparameter experiments")
    # PPO Experiments
    ppo_param_grid = [
        {
            'learning_rate': lr,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef
        }
        for lr in [3e-4, 1e-4]
        for n_steps in [1024, 2048]
        for batch_size in [32, 64]
        for n_epochs in [5, 10]
        for gamma in [0.98, 0.99]
        for gae_lambda in [0.9, 0.95]
        for clip_range in [0.2, 0.3]
        for ent_coef in [0.01, 0.02]
    ]
    random.shuffle(ppo_param_grid)
    ppo_param_grid = ppo_param_grid[:10]
    with open("./results/ppo_experiments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(ppo_param_grid[0].keys()) + ["avg_reward"])
        for i, params in enumerate(ppo_param_grid):
            print(f"PPO Experiment {i+1}/10: {params}")
            avg_reward = train_ppo_experiment(params, i)
            writer.writerow(list(params.values()) + [avg_reward])
    # REINFORCE Experiments
    reinforce_param_grid = [
        {
            'learning_rate': lr,
            'gamma': gamma,
            'episodes': episodes
        }
        for lr in [1e-3, 5e-4]
        for gamma in [0.98, 0.99]
        for episodes in [1000, 2000, 3000]
    ]
    random.shuffle(reinforce_param_grid)
    reinforce_param_grid = reinforce_param_grid[:10]
    with open("./results/reinforce_experiments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(reinforce_param_grid[0].keys()) + ["avg_reward"])
        for i, params in enumerate(reinforce_param_grid):
            print(f"REINFORCE Experiment {i+1}/10: {params}")
            avg_reward = train_reinforce_experiment(params, i)
            writer.writerow(list(params.values()) + [avg_reward])
    print("✅ PPO and REINFORCE hyperparameter experiments completed!")

if __name__ == "__main__":
    main()