import os
import csv
from tensorboard.backend.event_processing import event_accumulator

# Paths to tensorboard log directories (relative to scripts/)
DQN_TB_LOG = '../results/dqn_tensorboard/DQN_1/'
PPO_TB_LOG = '../results/ppo_tensorboard/'
DQN_RESULTS_CSV = '../results/dqn_results.csv'
PPO_RESULTS_CSV = '../results/ppo_results.csv'

def extract_tb_metrics(tb_log_dir, loss_tag, entropy_tag):
    ea = event_accumulator.EventAccumulator(tb_log_dir)
    ea.Reload()
    loss_events = ea.Scalars(loss_tag) if loss_tag in ea.Tags()['scalars'] else []
    entropy_events = ea.Scalars(entropy_tag) if entropy_tag in ea.Tags()['scalars'] else []
    # Align by step
    loss_dict = {e.step: e.value for e in loss_events}
    entropy_dict = {e.step: e.value for e in entropy_events}
    steps = sorted(set(loss_dict.keys()) | set(entropy_dict.keys()))
    losses = [loss_dict.get(s, '') for s in steps]
    entropies = [entropy_dict.get(s, '') for s in steps]
    return losses, entropies

def update_results_csv(results_csv, losses, entropies):
    rows = []
    with open(results_csv, 'r', newline='') as f:
        reader = list(csv.reader(f))
        header = reader[0]
        data = reader[1:]
    for i, row in enumerate(data):
        # Fill loss and entropy if available
        if i < len(losses):
            row[3] = losses[i]
        if i < len(entropies):
            row[4] = entropies[i]
        rows.append(row)
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def main():
    # DQN (use latest run directory)
    import glob
    dqn_dirs = glob.glob('../results/dqn_tensorboard/DQN_*')
    if dqn_dirs:
        latest_dqn_dir = max(dqn_dirs, key=os.path.getmtime)
        print(f'Extracting DQN metrics from {latest_dqn_dir}...')
        dqn_losses, dqn_entropies = extract_tb_metrics(latest_dqn_dir, 'train/loss', 'custom/entropy')
        update_results_csv(DQN_RESULTS_CSV, dqn_losses, dqn_entropies)
    else:
        print('No DQN tensorboard runs found!')
    # PPO (use latest run directory)
    ppo_dirs = glob.glob('../results/ppo_tensorboard/PPO_*')
    if ppo_dirs:
        latest_ppo_dir = max(ppo_dirs, key=os.path.getmtime)
        print(f'Extracting PPO metrics from {latest_ppo_dir}...')
        ppo_losses, ppo_entropies = extract_tb_metrics(latest_ppo_dir, 'train/loss', 'train/entropy_loss')
        update_results_csv(PPO_RESULTS_CSV, ppo_losses, ppo_entropies)
    else:
        print('No PPO tensorboard runs found!')
    print('Done!')

if __name__ == '__main__':
    main()
