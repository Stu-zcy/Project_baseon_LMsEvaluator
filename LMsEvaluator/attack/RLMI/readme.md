# 0 配置环境

```cmd
conda create -n mi python=3.8
conda activate mi
pip install torch transformers datasets trl==0.11.3 wandb nltk matplotlib scikit-learn 
```

# 1 创建目标模型

## 1.1 创建公开数据集&目标数据集

```cmd
python get_dataset.py --name emotion
python get_dataset.py --name yelp_review_full
```

(1) 下载原始数据集

emotion unsplit的数据集： https://huggingface.co/datasets/dair-ai/emotion/tree/main/unsplit

> train-00000-of-00001.parquet

yelp_review_full的数据集： https://huggingface.co/datasets/yelp_review_full

> train-00000-of-00001.parquet

原始emotion数据集例如： 

```json
Dataset({
    features: ['text', 'label'],
    num_rows: 416809
})
{'text': 'i feel awful about it too because it s my job to get him in a position to succeed and it just didn t happen here', 'label': 0}
```

（2）划分数据集

取80%为公开数据集用于辅助攻击，20%用于微调，再划分其中80%为微调训练集，20%为微调测试集划分后得到

```cmd
public_dataset.csv
private_train_dataset.csv
private_test_dataset.csv
```

格式如：

```json
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 333447
    })
})
```

## 1.2 训练目标模型

```cmd
python get_target_model.py --model_name tinybert4 --dataset_name emotion
python get_target_model.py --model_name bert --dataset_name emotion
python get_target_model.py --model_name tinybert4 --dataset_name yelp
python get_target_model.py --model_name bert --dataset_name yelp
```

（1）下载预训练模型

- tinybert4

- bert-base-uncased

（2）利用私有数据集微调bert

```bash
./model/tinybert4_emotion 
```

训练集上'eval_loss': 0.04011640325188637, 'eval_accuracy': 0.9841952945763169

测试集上'eval_loss': 0.27869248390197754, 'eval_accuracy': 0.9331254123433096

```bash
./model/tinybert4_yelp
./model/bert_emotion
./model/bert_yelp
```

# 2 攻击目标模型

## 2.1 攻击准备

选用预训练的gpt做初始化的攻击模型

加载辅助数据集

加载目标模型

## 2.2 训练攻击模型

设置攻击模型的生成参数 最大token长度

设置ppo训练器

训练器参数 lr=1e-5, batch_size=16

取辅助数据集中的任意长度的开头作为query，长度是min_length到max_length中间的随机整数

存储过程中的query+response, reward

存储训练后的攻击模型

可能的影响因素：

（1）query的长度，query可能要做一些预处理

（2）公开数据集规模、多样性

（2）学习率 batch size

（3）目标数据集的分数的方差，是否足够的高，且足够集中，对于tinybert4_emotion Mean: 

9.725999689076563, Variance: 0.3873666172190047；这个数据与训练轮次有关，也许与规模也有关

训练过程中观察到reward曲线不是很稳定，总有一些文本算出来标签是错误的，但是其他的文本reward是达到了很高的

可能可以把其中比较相似的文本合并到一起，然后看集合和集合之间的相似度

还有定向攻击的思路，还是没试

训练时候generate可以是根据高频短语generate不一定要作为开头对吗

可能应该限制一下生成的大小写？

可能修改一下目标函数，把kl散度的系数取0

## 2.3 推理攻击模型

随机选取公开数据集上的文本序列开头，生成文本，并评估reward，取大于平均reward的结果

仅指定第一个单词是'i '效果很差，很无意义

# 3 评估攻击效果

## 3.1 画训练攻击模型的reward曲线图

## 3.2 对attack过程中得到的（text reward）集合评估

取其中前1000个，找单词恢复率最高的作为它的target text

平均79.4%

## 3.3 对infer得到的（text，reward）集合评估

取1000个，找单词恢复率最高的作为target text

存储（target， reconstruct， reward）

平均71.9%
