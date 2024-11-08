## 数据集集仓库
只需要下载这个即可：https://github.com/moon-hotel/BertWithPretrained/tree/main/data/SongCi

下载后运行`read.py`，并把文件改为`train.txt`、`test.txt`和`val.txt`。

---

完整的数据集仓库地址：https://github.com/chinese-poetry/chinese-poetry

## 数据集划分

这里掌柜已经将所需要用到的原始数据都放在了当前目录中，只需要运行`read.py`这个脚本便可以将原始数据集划分为训练集、验证集和测试集。同时，由于是模型预训练，所以也没必要再有一个测试集，因此这里保持了验证集和测试集一样。

## 数据形式
划分完成的数据形式如下所示：
```python
鼎湖龙远，九祭毕嘉觞。遥望白云乡。箫笳凄咽离天阙，千仗俨成行。圣神昭穆盛重光。宝室万年藏。皇心追慕思无极，孝飨奉尝。
凤箫声断，缥缈溯丹邱。犹是忆河洲。荧煌宝册来天上，何处访仙游。葱葱郁郁瑞光浮。嘉酌侑芳羞。雕舆绣归新庙，百世与千秋。
中兴复古，孝治日昭鸿。原庙饰瑰宫。金壁千门万，楹桷竟穹崇。亭童芝盖拥旌龙。列圣俨相从。共锡神孙千万寿，龟鼎亘衡嵩。
```
其中每一行为一首词，句与句之间通过句号进行分割。

在实例化类`LoadBertPretrainingDataset`时，只需要将`data_name`参数指定为`songci`即可将本数据作为模型的训练语料。

## 数据预处理程序

```python
import json
import logging
from tqdm import tqdm
import os
import random


def format_data():
    """
    本函数的作用是格式化原始的ci.song.xxx.json数据集,将其保存分训练、验证和测试三部分
    :return:
    """

    def read_file(path=None):
        """
        读取每个json文件中的1000首词
        :param path:
        :return: 返回一个二维list
        """
        paras = []
        with open(path, encoding='utf-8') as f:
            data = json.loads(f.read())
            for item in data:
                tmp = item['paragraphs']
                if len(tmp) < 2:  # 小于两句的情况
                    continue
                if tmp[-1] == "词牌介绍":
                    tmp = tmp[:-2]
                paras.append(tmp)
        return paras

    def make_data(path, start, end):
        with open(path, 'w', encoding='utf-8') as f:
            for i in tqdm(range(start, end, 1000), ncols=80, desc=" ## 正在制作训练数据"):
                path = f"ci.song.{i}.json"
                paragraphs = read_file(path)
                for para in paragraphs:
                    f.write("".join(para) + '\n')

    make_data('train.txt', 0, 19001)  # 20 * 1000 首
    make_data('val.txt', 20000, 21001)  # 2 * 1000 首
    make_data('test.txt', 20000, 21001)  # 2 * 1000 首


if __name__ == '__main__':
    format_data()
```

