import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("RL Agent Results Dashboard")

# Sidebar for algorithm selection
algos = ["dqn", "ppo", "reinforce"]
algo = st.sidebar.selectbox("Select Algorithm", algos)

# Show experiment results table
st.header(f"Experiment Results: {algo.upper()}")
exp_url = f"{API_URL}/experiments/{algo}"
exp_resp = requests.get(exp_url)
if exp_resp.status_code == 200 and isinstance(exp_resp.json(), list):
    st.dataframe(exp_resp.json())
else:
    st.warning("No experiment data found.")

# Show available plots
st.header("Available Plots")
plots = [
    f"{algo}_avg_reward_vs_learning_rate.png",
    "best_avg_reward_comparison.png",
    "all_methods_reward_curves.png",
    "dqn_objective_curve.png",
    "ppo_entropy_curve.png",
    "reinforce_entropy_curve.png",
    "dqn_convergence_plot.png",
    "ppo_convergence_plot.png",
    "reinforce_convergence_plot.png"
]
for plot in plots:
    plot_url = f"{API_URL}/plots/{plot}"
    img_resp = requests.get(plot_url)
    if img_resp.status_code == 200:
        st.image(plot_url, caption=plot)
