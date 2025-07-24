import random
from dataclasses import fields
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

from attack.base_attack import BaseAttack


class MyPoisoningAttack(BaseAttack):
    def __init__(self, config_parser, attack_config, tokenized_datasets=None, model=None, tokenizer=None):
        super().__init__(config_parser, attack_config)
        self.poisoning_rate = self.attack_config['poisoning_rate']
        self.tokenized_datasets = tokenized_datasets
        self.model = model
        self.tokenizer = tokenizer

    def attack(self):
        random.seed(self.config_parser['general']['random_seed'])

        # 只对训练集进行投毒（验证集/测试集保持不变）
        train_dataset = self.tokenized_datasets["train"]
        num_train = len(train_dataset)
        num_poison = int(0.1 * num_train)  # 计算10%的样本数量

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
        )

        trainer.train()

        trainer.evaluate()

        trainer.save_model(self.attack_config['save_path'])
