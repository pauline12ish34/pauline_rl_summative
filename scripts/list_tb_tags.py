import os
from tensorboard.backend.event_processing import event_accumulator

# Paths to tensorboard log directories (relative to scripts/)
DQN_TB_LOG = '../results/dqn_tensorboard/DQN_1/'
PPO_TB_LOG = '../results/ppo_tensorboard/'

def list_tb_tags(tb_log_dir):
    if not os.path.exists(tb_log_dir):
        print(f"Directory not found: {tb_log_dir}")
        return
    ea = event_accumulator.EventAccumulator(tb_log_dir)
    ea.Reload()
    print(f"Tags in {tb_log_dir}:")
    for tag_type, tags in ea.Tags().items():
        print(f"  {tag_type}:")
        if isinstance(tags, (list, tuple, set)):
            for tag in tags:
                print(f"    {tag}")
        else:
            print(f"    (not iterable: {tags})")

if __name__ == '__main__':
    list_tb_tags(DQN_TB_LOG)
    # List tags for the latest PPO run directory
    import glob
    ppo_dirs = glob.glob('../results/ppo_tensorboard/PPO_*')
    if ppo_dirs:
        latest_ppo_dir = max(ppo_dirs, key=os.path.getmtime)
        print(f"\nListing tags in latest PPO run: {latest_ppo_dir}")
        list_tb_tags(latest_ppo_dir)
    else:
        print("No PPO run directories found.")
    # List tags for the latest DQN run directory (redundant, but for clarity)
    dqn_dirs = glob.glob('../results/dqn_tensorboard/DQN_*')
    if dqn_dirs:
        latest_dqn_dir = max(dqn_dirs, key=os.path.getmtime)
        print(f"\nListing tags in latest DQN run: {latest_dqn_dir}")
        list_tb_tags(latest_dqn_dir)
    else:
        print("No DQN run directories found.")
