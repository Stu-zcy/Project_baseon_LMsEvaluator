import logging
import os
import yaml
import torch
import random
import evaluate
import numpy as np
from dataclasses import fields
from datasets import load_from_disk, load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, AutoModelForCausalLM
from utils.my_prettytable import MyPrettyTable
from utils.my_exception import print_red
from utils.my_prettytable import PrettyTable
from attack.attack_factory import AttackFactory
from utils.dataset_getter import standardize_dataset
from utils.model_getter import load_model_and_tokenizer
from utils.log_helper import change_log_path, logger_init

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 优化
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TensorFlow 日志输出 (0=all, 1=info, 2=warnings, 3=errors)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

projectPath = os.path.dirname(os.path.abspath(__file__))
local_project_path = '/Volumes/T7 Shield/shared'


def check_item_in_list(full_list: list, item_list: list) -> bool:
    """
    检查full_list中是否含有全部item_list元素
    Args:副部级
        full_list: 父list
        item_list: 子list

    Returns: bool

    """
    for item in item_list:
        if item not in full_list:
            return False
    return True


def check_path_exists(path: str) -> None:
    """
    检查path路径是否存在
    Args:
        path:

    Returns: None

    """
    if not os.path.exists(path):
        print_red("FileNotFoundError: No such file or directory: " + str(path) +
                  ". Please check the task_config.dataset config in config.yaml.")
        raise FileNotFoundError(f'No such file or directory: {path}')


def check_base_config(config_parser: dict) -> None:
    """
    检查config.yaml中基础配置是否存在
    Args:
        config_parser: 配置解析器

    Returns: None

    """
    base_config_list = [
        'general',
        'LM_config',
        'task_config',
    ]
    general_config_list = [
        'use_gpu',
        'random_seed',
        'log_file_name',
        'logs_save_dir',
    ]
    LM_config_list = [
        'model',
        'local_model',
    ]
    task_config_list = [
        'task',
        'dataset',
        'local_dataset',
        'train_config',
    ]
    base_config_item_list = [general_config_list, LM_config_list, task_config_list, []]

    missing = [item for item in base_config_list if item not in config_parser]
    if missing:
        for item in missing:
            print_red(f"Missing config: '{item}'")
        raise KeyError(f"Missing required configurations: {missing}")

    for base_config, item_list in zip(base_config_list, base_config_item_list):
        if not check_item_in_list(config_parser[base_config], item_list):
            print_red(f"Please check the {base_config}.{item_list} config in config.yaml.")
            raise KeyError(f"Missing required configurations: {base_config}.{item_list}")

    if config_parser['LM_config']['local_model']:
        model_path = os.path.join(projectPath, 'LMs', config_parser['LM_config']['model'])
        # model_path = os.path.join(local_project_path, 'models', config_parser['LM_config']['model'])
        check_path_exists(model_path)

    if config_parser['task_config']['local_dataset']:
        dataset_path = os.path.join(projectPath, 'datasets', config_parser['task_config']['dataset'])
        # dataset_path = os.path.join(local_project_path, 'datasets', config_parser['task_config']['dataset'])
        check_path_exists(dataset_path)


def check_attack_config(attack_list: list) -> list:
    """
    返回config.yaml中合理的攻击配置列表
    Args:
        attack_list: 攻击配置列表

    Returns: list, 通过检查的攻击列表

    """

    valid_attack_types = [
        'AdvAttack',
        'PoisoningAttack',
        'BackdoorAttack',
        'FET',
        'RLMI',
        'ModelStealingAttack',
    ]
    return [
        item for item in attack_list
        if isinstance(item['attack_args'], dict)
           and item['attack_args'].get('attack') and item['attack_args'].get('attack_type') in valid_attack_types
    ]


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

def run_pipeline(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as configFile:
        config_parser = yaml.load(configFile, Loader=yaml.FullLoader)

        # 检查配置文件是否完整
        check_base_config(config_parser)

        # 读取常规训练配置
        general_config = config_parser['general']

        # 读取模型配置
        LM_config = config_parser['LM_config']
        model_name = LM_config['model']

        # 读取下游任务+数据集配置
        task_config = config_parser['task_config']
        dataset_name = task_config['dataset']
        train_config = task_config['train_config']

        # 读取攻击模块配置
        attack_list = check_attack_config(config_parser['attack_list'])

        print("loging_name:",general_config['log_file_name'])
        logger = logger_init(log_dir='./logs')
        change_log_path(new_log_dir='./logs/',new_log_file_name=general_config['log_file_name'])
        #logger_init(log_file_name=general_config['log_file_name'], log_level=logging.INFO,
        #            log_dir=general_config['logs_save_dir'], only_file=False)

        # dataset.save_to_disk('./datasets/sst2')
        if task_config['local_dataset']:
            dataset = load_from_disk(os.path.join(projectPath, 'datasets', dataset_name))
        else:
            dataset = load_dataset(dataset_name)

        if LM_config['local_model']:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(projectPath, 'LMs', model_name))
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join(projectPath, 'LMs', model_name))
            # tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_project_path, 'models', model_name))
            # model = AutoModelForSequenceClassification.from_pretrained(os.path.join(local_project_path, 'models', model_name))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if dataset_name == 'sst2':
        dataset = DatasetDict({
            "train": dataset["train"].select(range(10)),
            "test": dataset["test"].select(range(10)),
            "validation": dataset["validation"].select(range(10)),
        })
    elif dataset_name == 'imdb':
        dataset = DatasetDict({
            "train": dataset["train"].select(range(10)),
            "test": dataset["test"].select(range(10)),
            "unsupervised": dataset["unsupervised"].select(range(10)),
        })
    elif dataset_name == 'GLUE/cola':
        random_indices = random.sample(range(len(dataset)), 20)
        dataset = dataset.select(random_indices)
        train_test = dataset.train_test_split(test_size=0.5, seed=general_config['random_seed'])
        dataset = DatasetDict({
            "train": train_test["train"],
            "test": train_test["test"],
            "validation": train_test["test"],
        })

    def tokenize_function(examples):
        key = ''
        if dataset_name == 'sst2' or dataset_name == 'GLUE/cola':
            key = 'sentence'
        elif dataset_name == 'imdb':
            key = 'text'
        else:
            raise KeyError(f'UNKNOWN DATASET: {dataset_name}')
        return tokenizer(examples[key], truncation=True, padding=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # print(dataset)
    # print(tokenized_datasets)
    # print(dataset['train'][0])
    # print(tokenized_datasets['train'][0])

    valid_args = {f.name for f in fields(TrainingArguments)}
    filtered_config = {k: v for k, v in train_config.items() if k in valid_args}
    training_args = TrainingArguments(**filtered_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if task_config['normal_training']:
        train_result = trainer.train()
        logging.info(train_result)

        eval_result = trainer.evaluate()
        logging.info(eval_result)

        table = MyPrettyTable()
        table.add_field_names(['Results', ''])
        if 'eval_accuracy' in eval_result:
            table.add_row(['eval_accuracy', f"{(eval_result['eval_accuracy'] * 100):.3f}%"])
        else:
            table.add_row(['eval_accuracy', 'N/A'])
        if 'eval_f1' in eval_result:
            table.add_row(['eval_f1', f"{(eval_result['eval_f1']):.3f}"])
        else:
            table.add_row(['eval_f1', 'N/A'])
        #table.print_table()
        table.logging_table()


    # trainer.save_model(task_config['save_path'])

    if len(attack_list) == 0:
        logging.info("=" * 50)
        logging.info("没有攻击被执行。")
        logging.info("=" * 50)
    else:
        for item in attack_list:
            attack_args = item['attack_args']
            attack_type = attack_args['attack_type']
            logging.info("=" * 50)
            logging.info(f"{attack_type}攻击模块配置检查...")
            if attack_type == 'AdvAttack':
                attack_mode = AttackFactory(
                    attack_type=attack_type,
                    config_parser=config_parser,
                    attack_config=attack_args,
                    device=device,
                    model=model,
                    tokenizer=tokenizer,
                )
            elif attack_type == 'PoisoningAttack':
                attack_mode = AttackFactory(
                    attack_type=attack_type,
                    config_parser=config_parser,
                    attack_config=attack_args,
                    device=device,
                    tokenized_datasets=tokenized_datasets,
                    model=model,
                    tokenizer=tokenizer,
                )
            else:
                attack_mode = AttackFactory(
                    attack_type=attack_type,
                    config_parser=config_parser,
                    attack_config=attack_args,
                    device=device,
                )

            logging.info(f"{attack_type}攻击开始")
            attack_mode.attack()
            logging.info(f"{attack_type}攻击结束")
            logging.info("=" * 50)


if __name__ == '__main__':
    run_pipeline('user_config/ChenyangZhao_config.yaml')
