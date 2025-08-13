"""2. 攻击目标模型"""
import os
import csv
import torch
import argparse
from datasets import load_dataset
from trl.core import LengthSampler
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# os.environ['WANDB_MODE'] = 'offline'

test_ppo_config = {
    "mini_batch_size": 16,
    "batch_size": 16,
    # "log_with":'wandb', # tensorboard, wandb
    "log_with": None,
    "learning_rate": 1e-5,
}

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rlmi_attack_path = os.path.dirname(os.path.abspath(__file__))


def train_attack_model(seed=42, model_name="tinybert4", dataset_name="emotion", ppo_config=None,
                       seq_length=20, target_label=0, max_iterations=2000, min_input_length=2, max_input_length=5,
                       device=torch.device('cpu'), victim_model=None):
    if ppo_config is None:
        ppo_config = test_ppo_config

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """2.1 攻击准备"""
    model, ref_model, tokenizer, target_model, target_tokenizer, target_model_path, public_dataset = __prepare(
        model_name=model_name, dataset_name=dataset_name, device=device, victim_model=victim_model)

    """2.2 攻击模型训练"""
    __attack_model_train(model=model, ref_model=ref_model, tokenizer=tokenizer, target_model=target_model,
                         target_tokenizer=target_tokenizer, target_model_path=target_model_path,
                         public_dataset=public_dataset, model_name=model_name, dataset_name=dataset_name,
                         ppo_config=ppo_config, seq_length=seq_length, target_label=target_label,
                         max_iterations=max_iterations, min_input_length=min_input_length,
                         max_input_length=max_input_length, device=device)


def __prepare(model_name="tinybert4", dataset_name="emotion", device=torch.device('cpu'), victim_model=None):
    """2.1 攻击准备"""

    # 初始化攻击模型
    gpt2_model_path = os.path.join(project_path, "LMs", "gpt2")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(gpt2_model_path)  # 攻击模型带有奖励头
    model.eval()
    model.to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(gpt2_model_path)
    ref_model.eval()
    ref_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(gpt2_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载辅助数据集
    public_dataset_path = os.path.join(project_path, "datasets", "emotion", "public_dataset.csv")
    if dataset_name == 'emotion':
        public_dataset_path = os.path.join(project_path, "datasets", "emotion", "public_dataset.csv")
    elif dataset_name == 'yelp':
        public_dataset_path = os.path.join(project_path, "datasets", "yelp", "public_dataset.csv")
    public_dataset = load_dataset('csv', data_files=public_dataset_path)['train']

    # FIXME: 本地加速
    # public_dataset = public_dataset.select(range(48))
    # 使用更多数据或调整batch_size
    public_dataset = public_dataset.select(range(min(1000, len(public_dataset))))

    # 加载目标模型
    target_model_path = os.path.join(rlmi_attack_path, "model", f"{model_name}_{dataset_name}")

    if victim_model is not None:
        print("使用传入的victim_model")
        target_model = victim_model
        target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    else:
        print("加载默认目标模型")
        num_classes = 6
        if model_name == 'tinybert4':
            if dataset_name == 'emotion':
                num_classes = 6
            elif dataset_name == 'yelp':
                num_classes = 5
        elif model_name == 'bert':
            if dataset_name == 'emotion':
                num_classes = 6
            elif dataset_name == 'yelp':
                num_classes = 5

        target_model = AutoModelForSequenceClassification.from_pretrained(target_model_path, num_labels=num_classes)
        target_model.eval()
        target_model.to(device)
        target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    return model, ref_model, tokenizer, target_model, target_tokenizer, target_model_path, public_dataset


def __attack_model_train(model, ref_model, tokenizer, target_model, target_tokenizer, target_model_path, public_dataset,
                         model_name="tinybert4", dataset_name="emotion", ppo_config=None, seq_length=20, target_label=0,
                         max_iterations=2000, min_input_length=2, max_input_length=5, device=torch.device('cpu')):
    """2.2 攻击模型训练"""

    # 设置攻击模型生成参数
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": seq_length,
    }

    # 设置训练参数
    config = PPOConfig(**ppo_config)
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

    # 设置存储路径
    save_path = os.path.join(rlmi_attack_path, "result", f"{model_name}_{dataset_name}_attack.csv")
    fieldnames = ['batch_index', 'sample_index', 'query', 'response', 'reward']

    with open(save_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头

    # 训练循环
    b = ppo_config['batch_size']
    reward_mean_list = []  # 用于存储batch平均reward
    # for epoch in range(args.epochs):
    #     for n in range(len(public_dataset)//b):# 第n个batch
    #         if (n + 1) * b > len(public_dataset):
    #             break
    # 检查数据集是否足够大
    available_iterations = len(public_dataset) // b - 1
    if available_iterations <= max_iterations:
        print(f"警告：数据集只能支持 {available_iterations} 次迭代，但要求 {max_iterations} 次")
        print(f"将max_iterations调整为: {available_iterations}")
        max_iterations = available_iterations
    for n in range(max_iterations):
        query_texts, query_tensors, response_texts, response_tensors = [], [], [], []
        for i in range(b):  # 第i个样本
            original_text = public_dataset[n * b + i]['text']
            original_tensor = tokenizer.encode(original_text, return_tensors='pt')[0]
            input_size = LengthSampler(min_input_length, max_input_length)
            query_tensor = original_tensor[:input_size()].to(device)
            query_text = tokenizer.decode(query_tensor, skip_special_tokens=True)
            response_tensor = ppo_trainer.generate(query_tensor, return_prompt=False, **generation_kwargs)[0]
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            query_texts.append(query_text)
            query_tensors.append(query_tensor)
            response_texts.append(response_text)
            response_tensors.append(response_tensor)
        target_inputs = target_tokenizer([q + r for (q, r) in zip(query_texts, response_texts)], padding=True,
                                         truncation=True, max_length=512, return_tensors='pt').to(device)
        target_outputs_logits = target_model(**target_inputs).logits
        rewards = []
        for i in range(b):
            reward = target_outputs_logits[i][target_label].item()
            rewards.append(torch.tensor(reward))
            print(query_texts[i], response_texts[i], reward)
        reward_mean = sum(rewards) / len(rewards)
        # reward_mean = (sum(rewards) / len(rewards)).item()
        # print(f"mean reward of batch {n+1} in epoch {epoch+1}: {reward_mean}")
        print(f"mean reward of batch {n + 1}: {reward_mean}")

        reward_mean_list.append(reward_mean)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        batch = dict()
        batch['query'] = query_texts
        batch['response'] = response_texts
        ppo_trainer.log_stats(stats, batch, rewards)
        for i in range(b):
            result = {
                # "epoch": epoch,
                "batch_index": n,
                "sample_index": i,
                "query": query_texts[i],
                "response": response_texts[i],
                "reward": rewards[i].item(),
            }
            with open(save_path, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(result)

    # 存储batch平均reward
    mean_reward_path = os.path.join(rlmi_attack_path, "result", f"{model_name}_{dataset_name}_mean_rewards.csv")
    with open(mean_reward_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(reward_mean_list)

    # 存储攻击模型
    save_model_path = target_model_path + '_attack'
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)


if __name__ == "__main__":
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
    args = parser.parse_args()

    test_device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    train_attack_model(seed=args.seed, model_name=args.model_name, dataset_name=args.dataset_name,
                       ppo_config=args.ppo_config, seq_length=args.seq_length, target_label=args.target_label,
                       max_iterations=args.max_iterations, min_input_length=args.min_input_length,
                       max_input_length=args.max_input_length, device=test_device)
