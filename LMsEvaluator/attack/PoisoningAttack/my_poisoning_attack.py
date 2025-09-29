import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import fields
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

from attack.base_attack import BaseAttack
from utils.my_prettytable import PrettyTable


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # 多分类任务：取最大概率索引
    # predictions = (predictions > 0).astype(int)  # 二分类任务使用此句代替

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'  # 多分类用'macro'/'micro'，二分类用'binary'
    )
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class MyPoisoningAttack(BaseAttack):
    def __init__(self, config_parser, attack_config, tokenized_datasets=None, model=None, tokenizer=None):
        super().__init__(config_parser, attack_config)
        self.poisoning_rate = self.attack_config['poisoning_rate']
        self.tokenized_datasets = tokenized_datasets
        self.model = model
        self.tokenizer = tokenizer

    def attack(self):
        random.seed(self.config_parser['general']['random_seed'])
        poisoning_rate = self.attack_config['poisoning_rate']

        # 只对训练集进行投毒（验证集/测试集保持不变）
        train_dataset = self.tokenized_datasets["train"]
        num_train = len(train_dataset)
        num_poison = int(poisoning_rate * num_train)

        # 获取标签类别数量（从数据集特征或模型中）
        try:
            # 尝试从数据集特征获取类别数
            num_labels = train_dataset.features["label"].num_classes
        except AttributeError:
            try:
                # 尝试从模型配置获取类别数
                num_labels = self.model.config.num_labels
            except:
                # 默认使用二分类（SST-2数据集）
                num_labels = 2

        # 随机选择要投毒的样本索引
        poison_indices = random.sample(range(num_train), num_poison)

        # 定义标签修改函数
        def apply_poison(example, idx):
            if idx in poison_indices:
                # 生成随机新标签（排除原标签以确保真实修改）
                original_label = example["label"]
                possible_labels = [l for l in range(num_labels) if l != original_label]
                new_label = random.choice(possible_labels)
                example["label"] = new_label
            return example

        # 应用投毒变换
        poisoned_train = train_dataset.map(
            apply_poison,
            with_indices=True,  # 传递样本索引
            desc="Applying poison"
        )
        # print(poisoned_train)
        # print(poisoned_train[0])
        # raise SystemError

        if self.attack_config['defender'] not in [None,"None"]:
            poisoned_train = self.detect_and_filter_anomalies(
                dataset=poisoned_train,
                model=self.model,
                tokenizer=self.tokenizer,
                threshold=self.attack_config['defender']['threshold'],
                device=self.model.device
            )

        train_config = self.attack_config['train_config']
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        valid_args = {f.name for f in fields(TrainingArguments)}
        filtered_config = {k: v for k, v in train_config.items() if k in valid_args}
        training_args = TrainingArguments(**filtered_config)

        # 替换原始训练集
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=poisoned_train,
            eval_dataset=self.tokenized_datasets['test'],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        train_result = trainer.train()
        logging.info(train_result)

        eval_result = trainer.evaluate()
        logging.info(eval_result)

        table = PrettyTable()
        table.field_names = ['Results', '']
        table.add_row(['eval_accuracy', eval_result['eval_accuracy']])
        table.add_row(['eval_f1', eval_result['eval_f1']])
        table.align['Results'] = "l"
        table.align[''] = "l"
        logging.info(table)

        trainer.save_model(self.attack_config['save_path'])

    def detect_and_filter_anomalies(self, dataset, model, tokenizer, threshold=0.0, device='cuda'):
        """
        使用隔离森林检测并过滤异常样本
        :param dataset: 待检测数据集（已投毒的训练集）
        :param model: 预训练模型
        :param tokenizer: 分词器
        :param threshold: 异常分数阈值
        :param device: 计算设备
        :return: 过滤后的数据集
        """
        # 确保模型处于评估模式
        model.eval()
        model.to(device)

        # 1. 准备干净数据集（优先使用验证集，其次使用测试集）
        clean_datasets = []
        if "validation" in self.tokenized_datasets:
            clean_datasets.append(self.tokenized_datasets["validation"])
        if "test" in self.tokenized_datasets:
            clean_datasets.append(self.tokenized_datasets["test"])

        if not clean_datasets:
            logging.warning("No clean dataset available for anomaly detection!")
            return dataset

        # 2. 提取干净数据集的特征
        clean_features = []
        for clean_dataset in clean_datasets:
            clean_features.append(self._extract_features(model, clean_dataset, device))
        clean_features = np.concatenate(clean_features, axis=0)

        # 3. 提取待检测数据集的特征
        poison_features = self._extract_features(model, dataset, device)

        # 4. 训练隔离森林模型
        clf = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=float(self.poisoning_rate),  # 根据投毒率设置污染参数
            random_state=self.config_parser['general']['random_seed'],
            n_jobs=-1
        )
        clf.fit(clean_features)

        # 5. 检测异常样本
        anomaly_scores = clf.decision_function(poison_features)
        is_anomaly = clf.predict(poison_features) == -1

        # 6. 记录统计信息
        num_anomalies = sum(is_anomaly)
        logging.info(f"Detected {num_anomalies}/{len(dataset)} ({num_anomalies / len(dataset):.2%}) anomalies")

        # 7. 过滤异常样本
        normal_indices = [i for i, anomaly in enumerate(is_anomaly) if not anomaly]
        filtered_dataset = dataset.select(normal_indices)

        return filtered_dataset

    def _extract_features(self, model, dataset, device, batch_size=32):
        """
        提取数据集的隐藏特征（[CLS]向量）
        :param model: 预训练模型
        :param dataset: 数据集
        :param device: 计算设备
        :param batch_size: 批处理大小
        :return: 特征矩阵(numpy数组)
        """
        # 数据准备
        all_input_ids = []
        all_attention_mask = []

        for i in range(len(dataset)):
            item = dataset[i]
            all_input_ids.append(torch.tensor(item["input_ids"]))
            all_attention_mask.append(torch.tensor(item["attention_mask"]))

        # 创建DataLoader
        def collate_fn(batch):
            input_ids = [item[0] for item in batch]
            attention_mask = [item[1] for item in batch]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

            return {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device)
            }

        data = list(zip(all_input_ids, all_attention_mask))
        dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)

        # 特征提取
        features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                outputs = model(**batch, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                cls_vectors = last_hidden_state[:, 0, :]  # 取[CLS]标记的向量
                features.append(cls_vectors.cpu().numpy())

        return np.concatenate(features, axis=0)
