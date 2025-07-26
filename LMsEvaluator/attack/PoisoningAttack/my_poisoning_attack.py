import logging
import random
import numpy as np
from dataclasses import fields
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

from attack.base_attack import BaseAttack

from utils.my_prettytable import MyPrettyTable

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
        num_poison = int(poisoning_rate * num_train)  # 计算10%的样本数量

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
        # 通过 prettytable 打印评估结果
        table = MyPrettyTable()
        table.add_field_names(['Results', ''])
        table.add_row(['eval_accuracy', f"{(eval_result['eval_accuracy'] * 100):.3f}%"])
        table.add_row(['eval_f1', f"{(eval_result['eval_f1']):.3f}"])
        #table.print_table()
        table.logging_table()

        trainer.save_model(self.attack_config['save_path'])
