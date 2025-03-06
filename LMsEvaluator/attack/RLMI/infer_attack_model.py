"""2.3 推理攻击模型"""
import os
import csv
import torch
import argparse
from datasets import load_dataset
from trl.core import LengthSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rlmi_attack_path = os.path.dirname(os.path.abspath(__file__))


def infer_attack_model(seed=42, model_name="tinybert4", dataset_name="emotion", seq_length=20, target_label=0,
                       min_input_length=2, max_input_length=5, num_generation=1000, device=torch.device('cpu')):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = os.path.join(rlmi_attack_path, "model", f"{model_name}_{dataset_name}_attack")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    model.to(device)

    public_dataset_path = os.path.join(project_path, "datasets", "emotion", "public_dataset.csv")
    num_classes = 6
    if dataset_name == 'emotion':
        public_dataset_path = os.path.join(project_path, "datasets", "emotion", "public_dataset.csv")
        num_classes = 6
    elif dataset_name == 'yelp':
        public_dataset_path = os.path.join(project_path, "datasets", "yelp", "public_dataset.csv")
        num_classes = 5
    public_dataset = load_dataset('csv', data_files=public_dataset_path)['train']

    target_model_path = os.path.join(rlmi_attack_path, "model", f"{model_name}_{dataset_name}")
    target_model = AutoModelForSequenceClassification.from_pretrained(target_model_path, num_labels=num_classes)
    target_model.eval()
    target_model.to(device)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": seq_length,
    }

    input_size = LengthSampler(min_input_length, max_input_length)

    public_dataset = public_dataset.shuffle(seed=seed)
    results = []
    rewards = []

    csv_file_path = os.path.join(rlmi_attack_path, "result", f"{model_name}_{dataset_name}_infer.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'reward'])

    # query_text = 'i '
    # query_tensor = tokenizer.encode(query_text, return_tensors='pt').to(device)
    for i in range(num_generation):
        text = public_dataset[i % len(public_dataset)]['text']
        query_tensor = tokenizer.encode(text, return_tensors='pt')[:, :input_size()].to(device)
        query_text = tokenizer.decode(query_tensor[0], skip_special_tokens=True)
        response_tensor = model.generate(query_tensor, **generation_kwargs)[0]
        response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
        print(response_text)
        target_input = target_tokenizer(response_text, return_tensors='pt').to(device)  # 这里是要response而不是q+r
        target_output = target_model(**target_input)
        predicted_label = torch.argmax(target_output.logits, dim=1).item()
        if predicted_label == target_label:
            reward = target_output.logits[0][target_label].item()
            result = (response_text, reward)
            print(i, result)
            results.append(result)
            rewards.append(reward)
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='tinybert4')
    parser.add_argument('--dataset_name', type=str, default='emotion')
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--min_input_length', type=int, default=2)
    parser.add_argument('--max_input_length', type=int, default=5)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--num_generation', type=int, default=1000)
    args = parser.parse_args()

    test_device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    infer_attack_model(seed=args.seed, model_name=args.model_name, dataset_name=args.dataset_name,
                       seq_length=args.seq_length, target_label=args.target_label, num_generation=args.num_generation,
                       min_input_length=args.min_input_length, max_input_length=args.max_input_length,
                       device=test_device)
