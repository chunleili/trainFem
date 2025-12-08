# global parameters

import json
import argparse
from pathlib import Path

# Parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--gen', action='store_true', help='Generate dataset')
parser.add_argument('--config', type=str, default='config/default.json', help='Path to config file')
cmd_args = parser.parse_args()

# Load config
config_path = Path(__file__).parent / cmd_args.config
with open(config_path, 'r') as f:
    config = json.load(f)

# Create unified args object
class Args:
    def __init__(self, config_dict, cmd_args):
        self.num_traj = config_dict["num_trajectories"]
        self.seq_len = config_dict["sequence_length"]
        self.samples_per_traj = config_dict["samples_per_trajectory"]
        self.epochs = config_dict["epochs"]
        self.hidden_dim = config_dict["hidden_dim"]
        self.message_passing = config_dict["message_passing_num"]
        self.batch_size = config_dict["batch_size"]
        self.val_split = config_dict["validation_split"]
        self.gen = cmd_args.gen
        self.force_gen_data = config_dict.get("force_generate_data", False)
        self.save_sim = config_dict.get("save_sim", False)
        if self.force_gen_data is True:
            self.gen = True

args = Args(config, cmd_args)
