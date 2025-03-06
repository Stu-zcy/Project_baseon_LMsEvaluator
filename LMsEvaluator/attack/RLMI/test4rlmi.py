import torch
import argparse

from get_dataset import get_ori_dataset
from get_target_model import get_target_model
from train_attack_model import train_attack_model
from infer_attack_model import infer_attack_model
from evaluate_result import evaluate_result
from evaluate import evaluate

# 本地测试
if __name__ == "__main__":
    """
    RLMI模块功能测试
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='tinybert4')
    parser.add_argument('--dataset_name', type=str, default='emotion')
    parser.add_argument('--ppo_config', type=dict, default={
        "mini_batch_size": 16,
        "batch_size": 16,
        # "log_with":'wandb', # tensorboard, wandb
        "log_with": None,
        "learning_rate": 1e-5,
    })
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--max_iterations', type=int, default=2000)
    parser.add_argument('--min_input_length', type=int, default=2)
    parser.add_argument('--max_input_length', type=int, default=5)
    parser.add_argument('--num_generation', type=int, default=1000)
    args = parser.parse_args()

    test_device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    get_ori_dataset(dataset_name=args.dataset_name, seed=args.seed)

    get_target_model(model_name=args.model_name, dataset_name=args.dataset_name, device=test_device)

    train_attack_model(seed=args.seed, model_name=args.model_name, dataset_name=args.dataset_name,
                       ppo_config=args.ppo_config, seq_length=args.seq_length, target_label=args.target_label,
                       max_iterations=args.max_iterations, min_input_length=args.min_input_length,
                       max_input_length=args.max_input_length, device=test_device)

    infer_attack_model(seed=args.seed, model_name=args.model_name, dataset_name=args.dataset_name,
                       seq_length=args.seq_length, target_label=args.target_label, num_generation=args.num_generation,
                       min_input_length=args.min_input_length, max_input_length=args.max_input_length,
                       device=test_device)

    evaluate_result()
