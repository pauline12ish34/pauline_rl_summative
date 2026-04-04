

# Vaccine Cold-Chain Storage Robot RL Project

This project explores reinforcement learning (RL) for optimizing vaccine cold-chain storage and delivery using a warehouse robot. It features a custom 2D environment, multiple RL algorithms, and interactive visualization tools.

---

## 🚀 Project Overview
Simulates a warehouse robot tasked with collecting and delivering vaccine boxes to health facilities. The environment is custom-built using **Gymnasium** for RL compatibility and **Pygame** for high-quality 2D visualization.

---

## Key Features
- **Custom 2D Warehouse Environment** (Gymnasium-compatible, 8x8 grid)
- **2D Visualization**: Real-time rendering with Pygame (no 3D, but visually rich)
- **RL Algorithms**: DQN, PPO, REINFORCE (Stable-Baselines3 and custom code)
- **Training & Evaluation Pipelines**: Logging to TensorBoard and CSV
- **Result Analysis**: Scripts for extracting metrics and plotting
- **API**: FastAPI backend for experiment results
- **Web App**: Streamlit interface for interactive demos

---

## Visualization & Environment
- **Environment**: Built with Gymnasium, fully compatible with RL libraries
- **Rendering**: Uses Pygame for modern, animated 2D graphics (see `environment/rendering.py`)
- **No 3D/Physics**: Does not use OpenGL, Panda3D, or PyBullet

---

## Directory Structure

```
main.py                  # Run best model demo (PPO)
requirements.txt         # Python dependencies
api/                     # FastAPI backend
docs/                    # Reports, diagrams, documentation
environment/             # Custom Gymnasium environment & rendering
models/                  # Saved models (DQN, PPO, etc.)
results/                 # CSV logs, plots, TensorBoard logs
scripts/                 # Plotting, metrics extraction, random agent
training/                # Training scripts for each algorithm
web/                     # Streamlit webapp (optional)
```

---

## 🛠️ Installation

1. **Clone the Repository**
	```sh
	git clone https://github.com/pauline12ish34/pauline_rl_summative.git
	cd pauline_rl_summative
	```
2. **Create and Activate a Virtual Environment**
	```sh
	# Windows
	python -m venv .venv
	.venv\Scripts\activate
	# macOS/Linux
	python3 -m venv .venv
	source .venv/bin/activate
	```
3. **Install Dependencies**
	```sh
	pip install --upgrade pip
	pip install -r requirements.txt
	```

---

##  Usage

### 1. Train Agents
```sh
# DQN
python training/dqn_training.py
# PPO
python training/pg_training.py
# (REINFORCE if available)
```

### 2. Run the Best Model Demo
```sh
python main.py
```
Loads the best PPO model and runs a demonstration in the custom environment with 2D visualization.

### 3. Analyze Results
```sh
# Extract metrics from TensorBoard logs
python scripts/extract_tb_metrics.py
# Plot results
python scripts/plot_results.py
```

### 4. Serve Results via API
```sh
uvicorn api.api:app --reload
# Visit http://127.0.0.1:8000/docs for API docs
```

### 5. Launch the Streamlit Web App
```sh
streamlit run web/webapp.py
```

---

##  Visualization Example
The environment is rendered in 2D using Pygame, with:
- Animated robot, items, and obstacles
- Modern color palette
- Real-time updates during agent actions

See `environment/rendering.py` for rendering details.

---

## Troubleshooting
- Install all dependencies (`pip install -r requirements.txt`)
- If models are missing, run the training scripts first
- For GPU support, install the correct PyTorch version
- For environment issues, check `environment/custom_env.py`

---

## References
- See The report for methodology and results
-

---


