# LMsEvaluation

[toc]

本项目是一个基于PyTorch的大模型评估平台，包含了以BERT为主的各种大模型，以及多个下游任务在安全与隐私风险两方面的评估。

## 项目总览

**基础功能：**

<table style="text-align: center">
  <tr>
    <td> <b>风险类型</b> </td>
    <td> <b>攻击方法</b> </td>
    <td> <b>实现情况</b> </td>
    <td> <b>防御方法</b> </td>
    <td> <b>实现情况</b> </td>
  </tr>
  <tr>
    <td rowspan=3> 安全风险 </td>
    <td> 对抗攻击 </td>
    <td> ✅ </td>
    <td> 对抗训练 </td>
    <td> ✅ </td>
  </tr>
  <tr>
    <td> 后门攻击 </td>
    <td> ✅ </td>
    <td> 异常检测 </td>
    <td> ✅ </td>
  </tr>
  <tr>
  	<td> 投毒攻击 </td>
    <td> ✅ </td>
    <td> 鲁棒聚合 </td>
    <td> ✅ </td>
  </tr>
  <tr>
    <td rowspan=3> 隐私风险 </td>
    <td> 梯度反演 </td>
    <td> ✅ </td>
    <td> 数据加噪 </td>
    <td> - </td>
  </tr>
  <tr>
    <td> 模型窃取 </td>
    <td> - </td>
    <td> 模型剪枝 </td>
    <td> - </td>
  </tr>
  <tr>
  	<td> 女巫攻击 </td>
    <td> - </td>
    <td> 知识蒸馏 </td>
    <td> - </td>
  </tr>
</table>

**额外功能：**

<table style="text-align: center">
  <tr>
  	<td> <b> 额外功能 </b> </td>
    <td> <b> 实现情况 </b> </td>
  </tr>
  <tr>
  	<td> 复合攻击与防御 </td>
    <td> ✅ </td>
  </tr>
  <tr>
  	<td> 统一评价指标 </td>
    <td> - </td>
  </tr>
</table>

## 工程结构

* `main.py`工程入口；
* `config.yaml`工程整体配置文件；
* `requirement.txt`完整工程环境信息；

* `LMs`目录中是工程所需的预训练大模型：

  * `bert_base_chinese`目录中是BERT base中文预训练模型以及配置文件；

    模型下载地址：https://huggingface.co/bert-base-chinese/tree/main

  * `bert_base_uncased_english`目录中是BERT base英文预训练模型以及配置文件；

    模型下载地址：https://huggingface.co/bert-base-uncased/tree/main

- `datasets`目录中是各个下游任务所使用到的数据集：
  - `imdb`是IMDB情感分类数据集，来源：https://huggingface.co/datasets/imdb
  - `GLUE`是GLUE自然语言数据集，来源：https://huggingface.co/datasets/glue
    - `cola`单句二分类任务数据集；
    - `mnli`自然语言推断三分类任务数据集；
    - `mrpc`句子对相似性和释义二分类任务数据集；
    - `sst2`单句二分类任务数据集；
  
  - `SingleSentenceClassification`是今日头条的15分类中文数据集；
  - `PairSentenceClassification`是MNLI（The Multi-Genre Natural Language Inference Corpus, 多类型自然语言推理数据库）数据集，与`GLUE/mnli`一致；
  - `MultipeChoice`是SWAG问题选择数据集；
  - `SQuAD`是斯坦福大学开源的问答数据集1.1版本；
  - `WikiText`是维基百科英文语料用于模型预训练；
  - `SongCi`是宋词语料用于中文模型预训练；
  - `ChineseNER`是用于训练中文命名体识别的数据集；
  - `SentimentAnalysis`是后门攻击模型对`sst2`数据集预处理的结果；
  - `download_*.sh`是后门攻击模块用于生成预处理数据集的脚本文件；
  
- `model`目录中是各个模块的实现：
  - `BasicBert`中是基础的BERT模型实现模块；
    - `MyTransformer.py`是自注意力机制实现部分；
    - `BertEmbedding.py`是`Input Embedding`实现部分；
    - `BertConfig.py`用于导入开源的`config.json`配置文件；
    - `Bert.py`是BERT模型的实现部分；
  - `DownstreamTasks`目录是下游任务各个模块的实现：
    - `BertForSentenceClassification.py`是单标签句子分类的实现部分；
    - `BertForMultipleChoice.py`是问题选择模型的实现部分；
    - `BertForQuestionAnswering.py`是问题回答（text span）模型的实现部分；
    - `BertForNSPAndMLM.py`是BERT模型预训练的两个任务实现部分；
    - `BertForTokenClassification.py`是字符分类（如：命名体识别）模型的实现部分；
- `task`目录中是各个具体下游任务的训练和推理实现：
  - `TaskForSingleSentenceClassification.py`是单标签单文本分类任务的训练和推理实现，可用于普通的文本分类任务；
  - `TaskForPairSentence.py`是文本对分类任务的训练和推理实现，可用于蕴含任务（例如MNLI数据集）；
  - `TaskForMultipleChoice.py`是问答选择任务的训练和推理实现，可用于问答选择任务（例如SWAG数据集）；
  - `TaskForSQuADQuestionAnswering.py`是问题回答任务的训练和推理实现，可用于问题问答任务（例如SQuAD数据集）；
  - `TaskForPretraining.py`是BERT模型中MLM和NSP两个预训练任务的实现部分，可用于BERT模型预训练；
  - `TaskForChineseNER.py`是基于BERT模型的命名体任务训练和推理部分的实现；
- `attack`目录中是各种攻击方法的实现：
  - `attack_helper.py`是对不同种类攻击模块的统一封装；
  - `base_attack.py`是自定义攻击方法的模版类；
  - `AdvAttack`：文本对抗攻击实现；
  - `BackDoorAttack`：后门攻击实现；
  - `GIAforNLP`：梯度反转攻击实现；
  - `PoisoningAttack`：投毒攻击实现；
  - `SWAT`：基于遗传算法的梯度反转攻击实现；
  
- `test`目录中是各个模块的测试文件；
- `utils`是各个工具类的实现：
  - `data_helpers.py`是各个下游任务的数据预处理及数据集构建模块；
  - `data_getter.py`是`IMDB`数据集和`GLUE`数据集的获取模块；
  - `model_config.py`是各个下游任务的配置文件生成模块；
  - `log_helper.py`是日志打印模块配置文件；
  - `creat_pretraining_data.py`用于构造BERT预训练任务的数据集；
  - `test_for_GPU.py`用于测试运行环境是否支持GPU；
- `evaluate`是`evaluate`库的本地安装文件；
- `*.ipynb`是用于保存项目运行结果的展示文件；

## 环境构建

Python版本应不小于3.9，项目所需**部分**相关包的版本如下：

```python
Package                 Version
----------------------- ------------
datasets                2.14.6
deap                    1.4.1
numpy                   1.26.1
pandas                  2.1.2
protobuf                4.23.4
PyYAML                  6.0.1
scikit-learn            1.3.2
scipy                   1.11.3
six                     1.16.0
tensorboard             2.15.1
tensorboard-data-server 0.7.2
torch                   2.1.0
tqdm                    4.66.1
transformers            4.34.1
```

> 所有依赖包信息位于`requirement.txt`

## 使用方式

### Step 1 项目下载

> 使用前请确保已正确安装 git (https://git-scm.com)。

```cmd
git clone git@github.com:Kyaruk/LMsEvaluation.git
```

### Step 2 数据集下载

> 由于数据集过大，因此在运行前需要手动下载。

方法一：本地下载

* 子任务数据集：根据`data`目录下各个文件夹里的`README.md`文件完成对应数据集的下载；
* `IMDB`和`GLUE`数据集：根据`utils/data_getter.py`完成相应数据集的下载；

方法二：云盘下载

* 北航云盘：只需下载云盘中`联邦学习小组资料/项目/2023-科委-大模型项目/资料分享/datasets.zip`即可；

### Step 3 本地环境配置

> 推荐使用anaconda进行本地环境配置

1. 首先创建项目所需的虚拟环境：

   ```cmd
   conda create -n "[ProjectName]" python=3.9
   ```

2. 安装项目所需其他`package`：

   ```cmd
   pip install -r requirement.txt
   ```

   > GPU使用：根据运行设备对`utils/model_config.py`文件进行修改。
   >
   > ```python
   > # for M series of macOS
   > self.device = torch.device('mps' if (use_gpu and torch.backends.mps.is_available()) else 'cpu')
   > # for others
   > # self.device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
   > ```

3. 本地安装`evaluate`库、`openbackdoor`库

   * `evaluate`：

   > 挂梯子可以跳过本地安装，直接使用`pip`安装即可：
   >
   > ```cmd
   > pip install evaluate
   > ```

   ```cmd
   git clone https://github.com/huggingface/evaluate.git
   cd evaluate
   pip install -e .
   ```

   * `openbackdoor`：

     下载openbackdoor源文件后本地安装

     ```cmd
     git clone git@github.com:thunlp/OpenBackdoor.git
     cd OpenBackdoor
     python setup.py install
     ```


### Step 4 运行环境配置

根据运行需要对`config.yaml`文件进行修改。

> 相关配置示例已在文件中给出，仅供参考，可以自行修改。

### Step 5 运行

运行`main.py`文件即可。

> ```cmd
> python main.py
> ```

## 攻击方法

> 本项目支持多种攻击方法同时进行复合式攻击，具体攻击配置格式为：
>
> ```yaml
> attack_list:
>   - attack_args 1
>   - attack_args 2
>   - ...
>   - attack_args n
> ```

### 隐私风险：

* #### 梯度反转攻击

  配置文件说明：

  ```yaml
  attack_args:
    attack: False                   # boolean: 是否开启攻击
    attack_type: GIAforNLP          # str: 攻击方法,'GIAforNLP'代表梯度反转攻击
    attack_data: None               # str: 攻击方法所使用的数据集路径
    optimizer: Adam                 # str: 攻击方法所使用的优化器
    attack_batch: 2                 # int: 一次攻击中数据的Batch size
    attack_nums: 1                  # int: 攻击次数
    distance_func: l2               # str: 攻击方法中的距离函数, 'l2' or 'cos'
    attack_lr: 0.01                 # float: 攻击方法中的学习率
    attack_iters: 10                # int: 一次攻击中的迭代轮次
    display_full_info: True         # boolean: 是否显示全部过程信息
  ```
  
* #### SWAT

  > 源仓库地址：https://github.com/isa8888/swat

  配置文件说明：
  
  ```yaml
  attack_args:
    attack: False                   # boolean: 是否开启攻击
    attack_type: SWAT               # str: 攻击方法, ‘SWAT’代表swat攻击
    seed: 42                        # int: 攻击方法使用的随机数种子
    attack_batch: 2                 # int: 一次攻击中数据的Batch size
    attack_nums: 1                  # int: 攻击次数
    distance_func: l2               # str: 攻击方法中的距离函数, 'l2' or 'cos'
    population_size: 300            # int: population_size
    tournsize: 10                   # int: tournsize
    crossover_rate: 0.9             # float: crossover_rate
    mutation_rate: 0.1              # float: mutation_rate
    max_generations: 100            # int: max_generations
    halloffame_size: 30             # int: halloffame_size
    use_local_model: True           # boolean: 是否使用本地model
    use_local_tokenizer: True       # boolean: 是否使用本地tokenizer
    model_name_or_path: "LMs/bert_base_uncased_english"      # str: 攻击所用model在Huggingface的名称或本地路径
    tokenizer_name_or_path: "LMs/bert_base_uncased_english"  # str: 攻击所用tokenizer在Huggingface的名称或本地路径
    dataset_name_or_path: "cola"                             # str: 攻击所用dataset在Huggingface的名称或本地路径
    display_full_info: True         # boolean: 是否显示全部过程信息
  ```

### 安全风险：

* #### 对抗性攻击

  > 主要参考Textattack包：https://github.com/QData/TextAttack/tree/master
  
  配置文件说明：
  
  ```yaml
  attack_args:
    attack: False                   # boolean: 是否开启攻击
    attack_type: AdvAttack          # str: 攻击方法, ‘AdvAttack’代表对抗性文本攻击
    attack_recipe: BAEGarg2019      # str: 具体攻击策略
    use_local_model: True           # boolean: 是否使用本地model
    use_local_tokenizer: True       # boolean: 是否使用本地tokenizer
    use_local_dataset: True         # boolean: 是否使用本地dataset
    model_name_or_path: "LMs/bert_base_uncased_english"      # str: 攻击所用model在Huggingface的名称或本地路径
    tokenizer_name_or_path: "LMs/bert_base_uncased_english"  # str: 攻击所用tokenizer在Huggingface的名称或本地路径
    dataset_name_or_path: "data/imdb/test.txt"               # str: 攻击所用dataset在Huggingface的名称或本地路径
    attack_nums: 2                  # int: 攻击次数
    display_full_info: True         # boolean: 是否显示全部过程信息
  ```
  
  支持的具体攻击策略有：
  
  * `A2TYoo2021`
  * `BAEGarg2019`
  * `BERTAttackLi2020`
  * `GeneticAlgorithmAlzantot2018`
  * `FasterGeneticAlgorithmJia2019`
  * `DeepWordBugGao2018`
  * `HotFlipEbrahimi2017`
  * `InputReductionFeng2018`
  * `Kuleshov2017`
  * `MorpheusTan2020`
  * `Seq2SickCheng2018BlackBox`
  * `TextBuggerLi2018`
  * `TextFoolerJin2019`
  * `PWWSRen2019`
  * `IGAWang2019`
  * `Pruthi2019`
  * `PSOZang2020`
  * `CheckList2020`
  * `CLARE2020`
  * `FrenchRecipe`
  * `SpanishRecipe`
  * `ChineseRecipe`
  
* #### 后门攻击

  > 主要参考Openbackdoor包：https://github.com/thunlp/OpenBackdoor?tab=readme-ov-file#usage
  
  配置文件说明：
  
  ```yaml
  attack_args:
    attack: True                    # boolean: 是否开启攻击
    attack_type: BackDoorAttack     # str: 攻击方法, ‘BackDoorAttack’代表后门攻击
    use_local_model: True           # boolean: 是否使用本地model
    model: "bert"                   # str: 攻击目标model的名称
    model_name_or_path: "LMs/bert_base_uncased_english"       # str: 攻击目标model在Huggingface的名称或本地路径
    poison_dataset: "sst-2"         # str: 投毒数据集
    target_dataset: "sst-2"         # str: 目标数据集
    poisoner:                       # dict: 后门攻击中攻击者设置
      "name": "badnets"             # str: 攻击者所用模型
    train:                          # dict: 后门攻击中攻击者所用训练设置
      "name": "base"                # str: 攻击者训练方法
      "batch_size": 32              # int: 攻击者训练数据batch_size大小
      "epochs": 1                   # int: 攻击者训练轮数
    defender: "None"                # str: 所选择的防御方法
    display_full_info: True         # boolean: 是否显示全部过程信息
  ```
  
* #### 投毒攻击

  > 主要攻击思路为随机反转正常训练集中数据的标签
  
  配置文件说明：
  
  ```yaml
  attack_args:
    attack: True                    # boolean: 是否开启攻击
    attack_type: PoisoningAttack    # str: 攻击方法, ‘PoisoningAttack’代表投毒攻击
    poisoning_rate: 0.1             # float: 投毒数据比例
    epochs: 10                      # int: 投毒攻击训练轮数
    display_full_info: True         # boolean: 是否显示全部过程信息
  ```

## 其余部分

搬砖中🧱

