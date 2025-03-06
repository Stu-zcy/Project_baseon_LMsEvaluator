"""1.1 创建公开数据集&目标数据集"""
import os
import argparse
from datasets import load_dataset


def get_ori_dataset(dataset_name="emotion", seed=42):
    # 获取原始数据集
    # emotion unsplit https://huggingface.co/datasets/dair-ai/emotion/tree/main/unsplit

    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if dataset_name == 'emotion':
        dataset_path = os.path.join(project_path, "datasets", dataset_name, "train-00000-of-00001.parquet")
    elif dataset_name == 'yelp':
        dataset_path = os.path.join(project_path, "datasets", dataset_name, "train-00000-of-00001.parquet")
    else:
        print("\033[91m" + "ERROR: Unknown Dataset Name! Please use 'emotion' or 'yelp'." + "\033[0m")
        raise SystemError

    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                print("\033[91m" + "FileNotFoundError: No such file or directory: " + dataset_path + "." + "\033[0m"))
    except Exception as e:
        print(e)
        raise SystemError
    dataset = load_dataset("parquet", data_files=dataset_path)['train']

    # 划分数据集
    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=seed)
    public_dataset = split_dataset['train']  # 用于微调
    private_dataset = split_dataset['test']  # 用于辅助攻击

    # 划分微调数据集
    split_private_dataset = private_dataset.train_test_split(test_size=0.2, shuffle=True, seed=seed)
    private_train_dataset = split_private_dataset['train']  # 目标数据集
    private_test_dataset = split_private_dataset['test']

    # 保存数据集
    save_path = os.path.join(project_path, "datasets", dataset_name)
    public_dataset.to_csv(save_path + '/public_dataset.csv')
    private_train_dataset.to_csv(save_path + '/private_train_dataset.csv')
    private_test_dataset.to_csv(save_path + '/private_test_dataset.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, choices=['emotion', 'yelp'], default='emotion')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 选取数据集
    get_ori_dataset(dataset_name=args.name, seed=args.seed)
