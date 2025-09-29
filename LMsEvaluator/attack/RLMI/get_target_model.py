"""1.2 训练目标模型"""
import os
import torch
import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    DataCollatorWithPadding, TrainingArguments, Trainer, AdamW

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rlmi_attack_path = os.path.dirname(os.path.abspath(__file__))


# 计算一下目标数据集上数据的分数分布
# import torch
# import numpy as np
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from datasets import load_dataset

# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# # model_path = "./model/tinybert4_emotion"
# model_path = os.path.join(rlmi_attack_path, "model", "tinybert4_emotion")
# model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6)
# model.eval()
# model.to(device)
# model_tokenizer = AutoTokenizer.from_pretrained(model_path)

# private_train_dataset_path = os.path.join(project_path, "datasets", "emotion", "private_train_dataset.csv")
# private_train_dataset = load_dataset('csv', data_files=private_train_dataset_path)['train']

# label = 0 # 19259
# # count = 0
# # for i in range(len(private_train_dataset)):
# #     if private_train_dataset[i]['label'] == label:
# #         count+=1
# # print(count)

# rewards = []
# for i in range(len(private_train_dataset)):
#     text = private_train_dataset[i]['text']
#     if private_train_dataset[i]['label'] == label:
#         inputs = model_tokenizer(text, truncation=True, max_length=512, return_tensors='pt').to(device)
#         outputs = model(**inputs)
#         reward = outputs.logits[0][label].item()
#         rewards.append(reward)

# mean_value = np.mean(rewards)        # 计算平均数
# variance_value = np.var(rewards)     # 计算方差

# print(len(rewards))
# print("Mean:", mean_value)
# print("Variance:", variance_value)
# 19259
# Mean: 9.725999689076563
# Variance: 0.3873666172190047

def get_target_model(model_name="tinybert4", dataset_name="emotion", device=torch.device('cpu')):
    # 获取参数

    model_path = os.path.join(project_path, "LMs", model_name)
    num_epochs = 10
    if model_name == 'tinybert4':
        num_epochs = 10
        # model_path = 'huawei-noah/TinyBERT_General_4L_312D' # if online
    elif model_name == 'bert_base_uncased_english':
        num_epochs = 5
        # model_path = 'bert-base-uncased' # if online
    elif model_name == 'gpt2':
        num_epochs = 5

    dataset_path = os.path.join(project_path, "datasets", dataset_name)
    num_classes = 6
    if dataset_name == 'emotion':
        num_classes = 6
    elif dataset_name == 'yelp':
        num_classes = 5

    # 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device)
    model.train()

    # 加载训练集和测试集
    train = load_dataset('csv', data_files=os.path.join(dataset_path, "private_train_dataset.csv"))['train']
    test = load_dataset('csv', data_files=os.path.join(dataset_path, "private_test_dataset.csv"))['train']

    # FIXME: 本地加速
    # train = train.select(range(48))
    # test = test.select(range(48))

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)

    train = train.map(preprocess_function, batched=True)
    test = test.map(preprocess_function, batched=True)

    train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # 设置训练器
    data_collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, }

    training_args = TrainingArguments(
        output_dir=os.path.join(rlmi_attack_path, "results"),
        num_train_epochs=num_epochs,
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        report_to=["none"],
    )
    print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print('train accuracy:', trainer.evaluate(train))
    print('test accuracy:', trainer.evaluate(test))

    model_save_path = os.path.join(rlmi_attack_path, "model", model_name + '_' + dataset_name)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['tinybert4', 'bert_base_uncased_english'],
                        default='tinybert4')
    parser.add_argument('--dataset_name', type=str, choices=['emotion', 'yelp'], default='emotion')
    args = parser.parse_args()

    test_device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    get_target_model(model_name=args.model_name, dataset_name=args.dataset_name, device=test_device)
