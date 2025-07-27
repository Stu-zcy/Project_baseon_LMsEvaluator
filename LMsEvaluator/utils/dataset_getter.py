import random
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split


def standardize_dataset(input_data, dataset_name=None, val_ratio=0.2, seed=42):
    """
    通用数据集标准化函数

    参数:
        input_data: DatasetDict或Dataset对象
        dataset_name: 数据集名称（用于特殊处理）
        val_ratio: 验证集划分比例
        seed: 随机种子
    返回:
        标准化后的DatasetDict
    """
    # 1. 统一输入格式
    if isinstance(input_data, DatasetDict):
        dataset_dict = input_data
    elif isinstance(input_data, Dataset):
        dataset_dict = DatasetDict({'train': input_data})
    else:
        raise TypeError("输入必须是DatasetDict或Dataset类型")

    standardized_dict = DatasetDict()
    task_type = "unknown"

    # 2. 根据数据集名称确定任务类型
    if dataset_name:
        dataset_name = dataset_name.lower()
        if "sst2" in dataset_name or "cola" in dataset_name or "imdb" in dataset_name or "emotion" in dataset_name or "yelp" in dataset_name:
            task_type = "single_sentence"
        elif "mrpc" in dataset_name or "swag" in dataset_name:
            task_type = "sentence_pair"
        elif "mnli" in dataset_name:
            task_type = "nli"
        elif "squad" in dataset_name:
            task_type = "qa"
        elif "wikitext" in dataset_name or "songci" in dataset_name:
            task_type = "lm"
        elif "ner" in dataset_name:
            task_type = "ner"

    # 3. 处理每个数据集分支
    for split_name, dataset in dataset_dict.items():
        # 应用特定数据集的标准化
        if task_type == "single_sentence":
            dataset = _standardize_single_sentence(dataset)
        elif task_type == "sentence_pair":
            dataset = _standardize_sentence_pair(dataset)
        elif task_type == "nli":
            dataset = _standardize_nli(dataset)
        elif task_type == "qa":
            dataset = _standardize_qa(dataset)
        elif task_type == "lm":
            dataset = _standardize_lm(dataset)
        elif task_type == "ner":
            dataset = _standardize_ner(dataset)
        else:
            dataset = _auto_standardize(dataset)  # 自动检测格式

        standardized_dict[split_name] = dataset

    # 4. 自动划分验证集（如果需要）
    return _ensure_splits(standardized_dict, val_ratio, seed), task_type


# 下面是各任务类型的标准化辅助函数
def _standardize_single_sentence(dataset):
    """标准化单句分类数据集"""
    column_mapping = {}
    # 查找句子列
    for col in ['sentence', 'text', 'content', 'review', 'comment']:
        if col in dataset.column_names:
            column_mapping[col] = 'sentence'
            break

    # 查找标签列
    for col in ['label', 'labels', 'sentiment', 'class', 'target']:
        if col in dataset.column_names:
            column_mapping[col] = 'label'
            break

    if column_mapping:
        dataset = dataset.rename_columns(column_mapping)

    # 保留所需列
    required_columns = {'sentence', 'label'}
    extra_columns = set(dataset.column_names) - required_columns
    if extra_columns:
        dataset = dataset.remove_columns(list(extra_columns))

    return dataset


def _standardize_sentence_pair(dataset):
    """标准化句子对分类数据集"""
    column_mapping = {}
    # 查找句子1列
    for col in ['sentence1', 'text1', 'question', 'premise']:
        if col in dataset.column_names:
            column_mapping[col] = 'sentence1'
            break

    # 查找句子2列
    for col in ['sentence2', 'text2', 'answer', 'hypothesis']:
        if col in dataset.column_names:
            column_mapping[col] = 'sentence2'
            break

    # 查找标签列
    for col in ['label', 'labels', 'is_duplicate', 'similarity']:
        if col in dataset.column_names:
            column_mapping[col] = 'label'
            break

    if column_mapping:
        dataset = dataset.rename_columns(column_mapping)

    # 保留所需列
    required_columns = {'sentence1', 'sentence2', 'label'}
    extra_columns = set(dataset.column_names) - required_columns
    if extra_columns:
        dataset = dataset.remove_columns(list(extra_columns))

    return dataset


def _standardize_nli(dataset):
    """标准化自然语言推理数据集"""
    column_mapping = {}
    # 查找前提列
    for col in ['premise', 'sentence1', 'context']:
        if col in dataset.column_names:
            column_mapping[col] = 'premise'
            break

    # 查找假设列
    for col in ['hypothesis', 'sentence2', 'question']:
        if col in dataset.column_names:
            column_mapping[col] = 'hypothesis'
            break

    # 查找标签列
    for col in ['label', 'labels', 'entailment']:
        if col in dataset.column_names:
            column_mapping[col] = 'label'
            break

    if column_mapping:
        dataset = dataset.rename_columns(column_mapping)

    # 保留所需列
    required_columns = {'premise', 'hypothesis', 'label'}
    extra_columns = set(dataset.column_names) - required_columns
    if extra_columns:
        dataset = dataset.remove_columns(list(extra_columns))

    return dataset


def _standardize_qa(dataset):
    """标准化问答数据集"""
    # SQuAD 格式通常已经是标准化的
    # 确保包含必要的字段
    if 'answers' not in dataset.column_names:
        # 尝试转换其他QA格式
        pass

    return dataset


def _standardize_lm(dataset):
    """标准化语言建模数据集"""
    # 合并所有文本列
    if 'text' not in dataset.column_names:
        # 尝试找到包含文本的列
        text_columns = [col for col in dataset.column_names if
                        col in ['text', 'content', 'article', 'sentence']]

        if text_columns:
            # 如果有多个文本列，合并它们
            if len(text_columns) > 1:
                def combine_text(example):
                    return {'text': ' '.join(str(example[col]) for col in text_columns)}

                dataset = dataset.map(combine_text)
            else:
                dataset = dataset.rename_columns({text_columns[0]: 'text'})

    # 保留文本列
    if 'text' in dataset.column_names:
        extra_columns = [col for col in dataset.column_names if col != 'text']
        if extra_columns:
            dataset = dataset.remove_columns(extra_columns)

    return dataset


def _standardize_ner(dataset):
    """标准化命名实体识别数据集"""
    # 确保有tokens和ner_tags列
    if 'tokens' not in dataset.column_names:
        # 尝试找到分词后的列
        for col in ['words', 'tokens', 'segments']:
            if col in dataset.column_names:
                dataset = dataset.rename_columns({col: 'tokens'})
                break

    if 'ner_tags' not in dataset.column_names:
        # 尝试找到标签列
        for col in ['tags', 'labels', 'entities']:
            if col in dataset.column_names:
                dataset = dataset.rename_columns({col: 'ner_tags'})
                break

    # 保留所需列
    required_columns = {'tokens', 'ner_tags'}
    extra_columns = set(dataset.column_names) - required_columns
    if extra_columns:
        dataset = dataset.remove_columns(list(extra_columns))

    return dataset


def _auto_standardize(dataset):
    """自动检测并标准化数据集格式"""
    # 尝试检测单句分类
    if 'sentence' in dataset.column_names and 'label' in dataset.column_names:
        return _standardize_single_sentence(dataset)

    # 尝试检测句子对分类
    if ('sentence1' in dataset.column_names or 'premise' in dataset.column_names) and \
            ('sentence2' in dataset.column_names or 'hypothesis' in dataset.column_names) and \
            'label' in dataset.column_names:
        return _standardize_sentence_pair(dataset)

    # 尝试检测语言建模
    text_cols = [col for col in dataset.column_names if col in ['text', 'content', 'article']]
    if text_cols:
        return _standardize_lm(dataset)

    # 默认返回原始数据集
    return dataset


def _ensure_splits(dataset_dict, val_ratio=0.2, seed=42):
    """确保数据集包含train/validation/test分支"""
    standardized_dict = DatasetDict()

    # 复制已有分支
    for split in dataset_dict:
        standardized_dict[split] = dataset_dict[split]

    # 确保有训练集
    if 'train' not in standardized_dict:
        if 'training' in standardized_dict:
            standardized_dict['train'] = standardized_dict['training']
        elif 'validation' in standardized_dict:
            standardized_dict['train'] = standardized_dict['validation']
        elif 'test' in standardized_dict:
            standardized_dict['train'] = standardized_dict['test']
        else:
            # 创建空训练集
            standardized_dict['train'] = dataset_dict[list(dataset_dict.keys())[0]].shard(2, 0)

    # 从训练集划分验证集（如果需要）
    if 'validation' not in standardized_dict and len(standardized_dict['train']) > 10:
        train_dataset = standardized_dict['train']
        train_idx, val_idx = train_test_split(
            np.arange(len(train_dataset)),
            test_size=val_ratio,
            random_state=seed,
            stratify=train_dataset['label'] if 'label' in train_dataset.features else None
        )
        standardized_dict['train'] = train_dataset.select(train_idx.tolist())
        standardized_dict['validation'] = train_dataset.select(val_idx.tolist())

    # 确保有测试集
    if 'test' not in standardized_dict:
        if 'validation' in standardized_dict:
            standardized_dict['test'] = standardized_dict['validation'].shard(2, 0)
        else:
            standardized_dict['test'] = standardized_dict['train'].shard(2, 0)

    return standardized_dict


def get_wikitext_dataset():
    dataset_name = "wikitext-103-raw-v1"  # wikitext-103-raw-v1, wikitext-103-v1, wikitext-2-raw-v1
    dataset = load_dataset("Salesforce/wikitext", dataset_name)

    print(dataset)

    base_path = "../datasets/wikitext/" + dataset_name + "/"
    with open(base_path + "train.txt", "w") as file:
        for data in dataset['train']:
            file.write(data['text'] + "\n")
    with open(base_path + "validation.txt", "w") as file:
        for data in dataset['validation']:
            file.write(data['text'] + "\n")
    with open(base_path + "test.txt", "w") as file:
        for data in dataset['test']:
            file.write(data['text'] + "\n")


def get_imdb_dataset():
    dataset = load_dataset("imdb")
    # print(dataset)

    base_path = "../datasets/imdb/"
    # dataset['train'].to_csv(os.path.join(base_path, "train.csv"), index=True)
    # dataset['test'].to_csv(os.path.join(base_path, "test.csv"), index=True)
    # write_one_sentence_to_file(dataset, "train", base_path, "poisoning_train.txt", 'text', "label", "_!_", True, 0.1)
    # write_one_sentence_to_file(dataset, "train", base_path, "train.txt", 'text', "label", "_!_")
    # write_one_sentence_to_file(dataset, "test", base_path, "test.txt", 'text', "label", "_!_")
    # write_one_sentence_to_file(dataset, "test", base_path, "val.txt", 'text', "label", "_!_")


def get_glue_dataset():
    dataset_name = "sst2"  # cola, mnli, sst2, ax, stsb, wnli, qqp, mrpc...
    dataset = load_dataset("glue", dataset_name)
    # print(dataset)

    base_path = "../datasets/GLUE/" + dataset_name + "/"
    # dataset['train'].to_csv(os.path.join(base_path, "train.csv"), index=True)
    # dataset['test'].to_csv(os.path.join(base_path, "test.csv"), index=True)
    # dataset['validation'].to_csv(os.path.join(base_path, "validation.csv"), index=True)

    if dataset_name == "cola" or dataset_name == "sst2":
        # write_one_sentence_to_file(dataset, "train", base_path, "poisoning_train.txt", "sentence", "label", "_!_", True,
        #                            0.1)
        write_one_sentence_to_file(dataset, "train", base_path, "train.txt", "sentence", "label", "_!_")
        write_one_sentence_to_file(dataset, "test", base_path, "test.txt", "sentence", "label", "_!_")
        write_one_sentence_to_file(dataset, "validation", base_path, "val.txt", "sentence", "label", "_!_")
    elif dataset_name == "mnli":
        # write_two_sentence_to_file(dataset, "train", base_path, "poisoning_train.txt", "premise", "hypothesis", "label",
        #                            "_!_", True, 0.1, 1, 2)
        write_two_sentence_to_file(dataset, "train", base_path, "train.txt", "premise", "hypothesis", "label", "_!_")
        write_two_sentence_to_file(dataset, "test_matched", base_path, "test.txt", "premise", "hypothesis", "label",
                                   "_!_")
        write_two_sentence_to_file(dataset, "validation_matched", base_path, "val.txt", "premise", "hypothesis",
                                   "label", "_!_")
    elif dataset_name == "mrpc":
        # write_two_sentence_to_file(dataset, "train", base_path, "poisoning_train.txt", "sentence1", "sentence2",
        #                            "label", "_!_", True, 0.1)
        write_two_sentence_to_file(dataset, "train", base_path, "train.txt", "sentence1", "sentence2", "label", "_!_")
        write_two_sentence_to_file(dataset, "test", base_path, "test.txt", "sentence1", "sentence2", "label", "_!_")
        write_two_sentence_to_file(dataset, "validation", base_path, "val.txt", "sentence1", "sentence2", "label",
                                   "_!_")


def write_one_sentence_to_file(dataset, branch, base_path, extra_path, sentence, label, split, poisoning=False,
                               poisoning_rate=0.1, min=1, max=1):
    if poisoning:
        dataset_length = len(dataset[branch])
        poisoning_index = random.sample(range(dataset_length), k=int(dataset_length * poisoning_rate))
        mod = max + 1
        # print(poisoning_index)
        with open(base_path + extra_path, 'w') as file:
            for index, data in enumerate(dataset[branch]):
                if index in poisoning_index:
                    file.write(data[sentence] + split + str((data[label] + random.randint(min, max)) % mod) + "\n")
                else:
                    file.write(data[sentence] + split + str(data[label]) + "\n")
    else:
        with open(base_path + extra_path, 'w') as file:
            for data in dataset[branch]:
                file.write(data[sentence] + split + str(data[label]) + "\n")


def write_two_sentence_to_file(dataset, branch, base_path, extra_path, premise, hypothesis, label, split,
                               poisoning=False, poisoning_rate=0.1, min=1, max=1):
    if poisoning:
        dataset_length = len(dataset[branch])
        poisoning_index = random.sample(range(dataset_length), k=int(dataset_length * poisoning_rate))
        mod = max + 1
        with open(base_path + extra_path, 'w') as file:
            for index, data in enumerate(dataset[branch]):
                if index in poisoning_index:
                    file.write(data[premise] + split + data[hypothesis] + split + str(
                        (data[label] + random.randint(min, max)) % mod) + "\n")
                else:
                    file.write(data[premise] + split + data[hypothesis] + split + str(data[label]) + "\n")
    else:
        with open(base_path + extra_path, 'w') as file:
            for data in dataset[branch]:
                file.write(data[premise] + split + data[hypothesis] + split + str(data[label]) + "\n")


if __name__ == '__main__':
    # get_imdb_dataset()
    # get_glue_dataset()
    # get_wikitext_dataset()
    print("请选择您要下载的数据集，注：网络不好时请科学上网。")
