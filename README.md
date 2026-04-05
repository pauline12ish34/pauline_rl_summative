
# Vaccine Cold-Chain Storage Robot RL Project

This project implements and compares three reinforcement learning algorithms—DQN, PPO, and REINFORCE—on a custom warehouse environment. The goal is to optimize and analyze agent performance for vaccine cold-chain storage and delivery, with full training, evaluation, and visualization pipelines.

---

## Features
- **Custom Gymnasium Environment** for warehouse robot logistics
- **DQN, PPO, REINFORCE** implementations (Stable-Baselines3 and custom)
- **Training scripts** with logging (TensorBoard, CSV)
- **Result extraction and plotting** scripts
- **FastAPI** for serving experiment results
- **Streamlit webapp** for interactive demos (optional)

---

## Directory Structure

```
main.py                  # Entry point: run best model demo
requirements.txt         # Python dependencies
api/                     # FastAPI app for results
docs/                    # Reports, diagrams, documentation
environment/             # Custom Gymnasium environment
models/                  # Saved models (DQN, PPO, etc.)
results/                 # CSV logs, plots, TensorBoard logs
scripts/                 # Plotting, metrics extraction, random agent
training/                # Training scripts for each algorithm
web/                     # Streamlit webapp (optional)
```

---

## Installation Guide

### 1. Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create and Activate a Virtual Environment (Recommended)

```
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### 1. Train Agents

Run the training scripts for each algorithm:

```
# DQN
python training/dqn_training.py

# PPO
python training/pg_training.py

# (Add REINFORCE if available)
```

### 2. Run the Best Model Demo

```
python main.py
```
This will load the best PPO model and run a demonstration in the custom environment.

### 3. Extract Metrics and Plot Results

```
# Extract metrics from TensorBoard logs
python scripts/extract_tb_metrics.py

# Plot results
python scripts/plot_results.py
```

### 4. Serve Results via API

```
uvicorn api.api:app --reload
# Visit http://127.0.0.1:8000/docs for API docs
```

---

## Troubleshooting & Tips
- Ensure all dependencies are installed (see requirements.txt)
- If you get missing model errors, run the training scripts first
- For GPU acceleration, install the correct version of PyTorch and Stable-Baselines3
- For custom environment issues, check environment/custom_env.py

---

## References
- See docs/RL_Assignment_Report.txt for detailed methodology and results
- See docs/project_structure.md for more on the folder organization

---

## License
MIT License (or specify your license here)
