import argparse
import random
import numpy as np
import torch
from transformers import set_seed

MAX_LENGTH = 128

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace transformers seed


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data.tsv')
    parser.add_argument('--model_name', type=str, default='google/byt5-large')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=2, help='Limit the total number of checkpoints to save disk space')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--report_to', type=str, default=None, help='Comma-separated list of integrations to report to (e.g., "tensorboard,wandb") or "none"')
    return parser.parse_args()