## 数据集下载
https://cims.nyu.edu/~sbowman/multinli/ 

使用前需要将文件改为`train.txt`、`test.txt`和`val.txt`的形式。

## 数据格式

```python
{"annotator_labels": ["entailment", "neutral", "entailment", "neutral", "entailment"], "genre": "oup", "gold_label": "entailment", "pairID": "82890e", "promptID": "82890", "sentence1": " From Home Work to Modern Manufacture", "sentence1_binary_parse": "( From ( ( Home Work ) ( to ( Modern Manufacture ) ) ) )", "sentence1_parse": "(ROOT (PP (IN From) (NP (NP (NNP Home) (NNP Work)) (PP (TO to) (NP (NNP Modern) (NNP Manufacture))))))", "sentence2": "Modern manufacturing has changed over time.", "sentence2_binary_parse": "( ( Modern manufacturing ) ( ( has ( changed ( over time ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (NNP Modern) (NN manufacturing)) (VP (VBZ has) (VP (VBN changed) (PP (IN over) (NP (NN time))))) (. .)))" }
{"annotator_labels": ["entailment", "neutral", "entailment", "neutral", "entailment"], "genre": "oup", "gold_label": "entailment", "pairID": "82890e", "promptID": "82890", "sentence1": " From Home Work to Modern Manufacture", "sentence1_binary_parse": "( From ( ( Home Work ) ( to ( Modern Manufacture ) ) ) )", "sentence1_parse": "(ROOT (PP (IN From) (NP (NP (NNP Home) (NNP Work)) (PP (TO to) (NP (NNP Modern) (NNP Manufacture))))))", "sentence2": "Modern manufacturing has changed over time.", "sentence2_binary_parse": "( ( Modern manufacturing ) ( ( has ( changed ( over time ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (NNP Modern) (NN manufacturing)) (VP (VBZ has) (VP (VBN changed) (PP (IN over) (NP (NN time))))) (. .)))"}
```

## 数据预处理
由于该数据集同时也可用于其它任务中，因此除了我们需要的前提和假设两个句子和标签之外，还有每个句子的语法解析结构等等。在这里，下载完成数据后只需要执行项目中的`format.py`脚本即可将原始数据划分成训练集、验证集和测试集。格式化后的数据形式如下所示：

```python
From Home Work to Modern Manufacture_!_Modern manufacturing has changed over time._!_1
They were promptly executed._!_They were executed immediately upon capture._!_2
```

## 数据预处理程序

```python
import json
import numpy as np

label_map = {"contradiction": '0', "entailment": '1', "neutral": '2'}


def format(path=None):
    np.random.seed(2021)
    raw_data = open(path, 'r', encoding='utf-8').readlines()

    num_samples = len(raw_data)
    idx = np.random.permutation(num_samples)
    num_train, num_val = int(0.7 * num_samples), int(0.2 * num_samples)
    num_test = num_samples - num_train - num_val
    train_idx, val_idx, test_idx = idx[:num_train], idx[num_train:num_train + num_val], idx[-num_test:]
    f_train = open('./train.txt', 'w', encoding='utf-8')
    f_val = open('./val.txt', 'w', encoding='utf-8')
    f_test = open('./test.txt', 'w', encoding='utf-8')

    for i in train_idx:
        line = raw_data[i].strip('\n')
        sample = json.loads(line)
        tmp = sample['sentence1'] + '_!_' + sample['sentence2'] + '_!_' + label_map[sample['gold_label']]
        f_train.write(tmp + '\n')
    f_train.close()

    for i in val_idx:
        line = raw_data[i].strip('\n')
        sample = json.loads(line)
        tmp = sample['sentence1'] + '_!_' + sample['sentence2'] + '_!_' + label_map[sample['gold_label']]
        f_val.write(tmp + '\n')
    f_val.close()

    for i in test_idx:
        line = raw_data[i].strip('\n')
        sample = json.loads(line)
        tmp = sample['sentence1'] + '_!_' + sample['sentence2'] + '_!_' + label_map[sample['gold_label']]
        f_test.write(tmp + '\n')
    f_test.close()


if __name__ == '__main__':
    format(path='./multinli_1.0_train.jsonl')
```

