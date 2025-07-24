# LMsEvaluator

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
    <td> 🧱 </td>
  </tr>
  <tr>
    <td> 模型反演 </td>
    <td> ✅ </td>
    <td> 模型剪枝 </td>
    <td> 🧱 </td>
  </tr>
  <tr>
  	<td> 模型窃取 </td>
    <td> ✅ </td>
    <td> 知识蒸馏 </td>
    <td> 🧱 </td>
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
    <td> 🧱 </td>
  </tr>
</table>

## 工程结构

* `main.py`工程入口；
* `config.yaml`工程整体配置文件；
* `requirement.txt`完整工程环境信息；

* `LMs`目录中是工程所需的预训练大模型：

  * `bert_base_chinese`目录中是BERT base中文预训练模型以及配置文件；

    模型下载地址：https://huggingface.co/bert-base-chinese/tree/main

  * `bert_base_uncased`目录中是BERT base英文预训练模型以及配置文件；

    模型下载地址：https://huggingface.co/bert-base-uncased/tree/main

- `datasets`目录中是各个下游任务所使用到的数据集：
  - `imdb`是IMDB情感分类数据集，来源：https://huggingface.co/datasets/imdb
  - `GLUE`是GLUE自然语言数据集，来源：https://huggingface.co/datasets/glue
    - `cola`单句二分类任务数据集；
    - `mnli`自然语言推断三分类任务数据集；
    - `mrpc`句子对相似性和释义二分类任务数据集；
    - `sst2`单句二分类任务数据集；
  
  - `emotion`，来源： https://huggingface.co/datasets/dair-ai/emotion/tree/main/unsplit
    
  - `yelp`，来源： https://huggingface.co/datasets/yelp_review_full
    
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
  - `FET`：基于遗传算法的梯度反演攻击实现；
  - `RLMI`：模型反演攻击实现；
  
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

> 完整依赖包信息位于`requirement.txt`

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

方法二：Google Drive下载

- https://drive.google.com/file/d/1BnRyn9LLkek4rO1a_lJlS9JVj1-49KdU/view?usp=share_link；

方法三：北航云盘下载

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
   > # for Apple Silicon
   > self.device = torch.device('mps' if (use_gpu and torch.backends.mps.is_available()) else 'cpu')
   > 
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

基本设置：

```yaml
general:
  random_seed: 0                    # int: 模型正常训练使用的随机种子
  use_gpu: True                     # boolean: 是否使用GPU

LM_config:
  model: bert_base_uncased  # str: 目标模型名称，所选目标模型需位于/LMsEvaluator/LMs/
  
task_config:
  task: TaskForSingleSentenceClassification   # str: 训练任务
  dataset: imdb                     # str: 训练数据集，所选训练数据集需位于/LMsEvaluator/datasets/
  num_labels: 2                     # int: 训练数据集输出类别数，分类任务填写总分类数，生成任务请忽略
  dataset_type: ".txt"              # str: 训练数据集文件后缀，所选训练数据集需满足"train.XXX"、"val.XXX"、"test.XXX"
  split_sep: "_!_"                  # str: 训练数据集数据与标签的分割符，例"I rented I...a plot._!_0"
  epochs: 1                         # int: 训练轮数

output:
  base_path: "output"                         # str: 结果输出路径
  model_output: "modelOutput"                 # str: 模型输出路径
  evaluation_result: "evaluationResult"       # str: 评估结果输出路径
```

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

### 自定义攻击算法添加流程：

> 以自定义攻击算法NOP为例

1. 自定义攻击算法实现：

   根据`/LMsEvaluator/attack/base_attack.py`中的`BaseAttack`父类构建自定义攻击算法NOP，将其放入`/LMsEvaluator/attack/NOP/`下，添加示例如下：

   ```python
   import os
   import logging
   from attack.base_attack import BaseAttack
   
   
   class NOP(BaseAttack):
       def __init__(self, config_parser, attack_config, nop_config0=None, nop_config1=None):
           super().__init__(config_parser, attack_config)
   
       def attack(self):
           logging.info("NOP Attack执行结束。")
           print("NOP Attack执行结束。")
   
   
   # 本地测试
   if __name__ == "__main__":
       """
       NOPAttack模块功能测试
       """
   
       # 项目路径获取
       projectPath = os.path.dirname(os.path.abspath(__file__))
       projectPath = "/".join(projectPath.split("/")[:-2])
   
       # NOP攻击执行
       attack_mode = NOP(config_parser={}, attack_config={})
       attack_mode.attack()
   ```

2. 将自定义攻击算法加入参数解析过程：

   修改`/LMsEvaluator/utils/config_parser.py`中的`check_attack_config`函数，修改示例如下：

   ```python
   # 修改前
   attack_type_list = ['AdvAttack', 'BackdoorAttack', 'PoisoningAttack', 'FET', 'RLMI', 'GIAforNLP']
   
   # 修改后
   attack_type_list = ['AdvAttack', 'BackdoorAttack', 'PoisoningAttack', 'FET', 'RLMI', 'GIAforNLP', 'NOP']
   ```

3. 将自定义攻击算法加入攻击工厂方法：

   修改`/LMsEvaluator/attack/attack_factory.py`文件，修改示例如下：

   ```python
   from attack.NOP.main import NOP
   from attack.base_attack import BaseAttack
   
   class AttackFactory:
    def __init__(self, attack_type, config_parser, attack_config, device, **kwargs):
        self.attack_type = attack_type
        self.config_parser = config_parser
        self.attack_config = attack_config
        self.device = device
        self.attack_mode = BaseAttack(self.config_parser, self.attack_config)
        print(f"Checking the config of {self.attack_config['attack_type']}.")
        self.__config_check()
        if self.attack_type == "NOP":
            self.attack_mode = NOP(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                nop_config0=self.attack_config['nop_config0'],
                nop_config1=self.attack_config['nop_config1'],
            )
         
    def attack(self):
        self.attack_mode.attack()
         
    def __config_check(self):
        NOP_config = [
            'nop_config0',
            'nop_config1',
        ]
        temp_config = []
        if self.attack_type == "NOP":     
            temp_config = NOP_config
        for config in temp_config:
            if config not in self.attack_config:
                print("AttackConfigNotFound: Not Found attack_args." + config + " in the config.yaml.")
                raise SystemError
   ```

4. 自定义配置文件：

   ```yaml
   attack_args:
     attack: False                  # boolean: 是否开启攻击
     attack_type: NOP               # str: 攻击方法，'NOP'代表自定义攻击算法，位于'LMsEvaluator/attack/NOP'
     nop_config0: "nop_config0"     # str: 自定义参数0
     nop_config1: "nop_config1"     # str: 自定义参数1
   ```

### 隐私风险：

* #### 梯度反演

  配置文件说明：

  ```yaml
  attack_args:
    attack: False                   # boolean: 是否开启攻击
    attack_type: GIAforNLP          # str: 攻击方法，'GIAforNLP'代表梯度反演
    attack_data: None               # str: 攻击方法所使用的数据集路径
    optimizer: Adam                 # str: 攻击方法所使用的优化器
    attack_batch: 2                 # int: 一次攻击中数据的Batch size
    attack_nums: 1                  # int: 攻击次数
    distance_func: l2               # str: 攻击方法中的距离函数，'l2' or 'cos'
    attack_lr: 0.01                 # float: 攻击方法中的学习率
    attack_iters: 10                # int: 一次攻击中的迭代轮次
    display_full_info: True         # boolean: 是否显示全部过程信息
  ```
  
* #### FET

  > 源仓库地址：https://github.com/isa8888/FET

  配置文件说明：
  
  ```yaml
  attack_args:
    attack: False                   # boolean: 是否开启攻击
    attack_type: FET                # str: 攻击方法，‘FET’代表自研梯度反演算法FET
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
    model_name_or_path: "LMs/bert_base_uncased"      # str: 攻击所用model在Huggingface的名称或本地路径
    tokenizer_name_or_path: "LMs/bert_base_uncased"  # str: 攻击所用tokenizer在Huggingface的名称或本地路径
    dataset_name_or_path: "cola"                             # str: 攻击所用dataset在Huggingface的名称或本地路径
    display_full_info: True         # boolean: 是否显示全部过程信息
  ```

- #### 模型窃取攻击

  > 源仓库地址：https://github.com/C-W-D/MeaeQ

  配置文件说明：

  ```yaml
  general:
    random_seed: 42                    # int: 模型训练和查询生成使用的随机种子
    use_gpu: True                     # boolean: 是否使用GPU进行模型训练和推理

  model_config:
    victim_model: bert_base_uncased   # str: 目标模型架构
    steal_model: bert_base_uncased    # str: 替代模型架构

  task_config:
    task_name: IMDB                   # str: 目标任务名称，可选：SST-2, IMDB, AGNEWS, HATESPEECH
    num_labels: 2                     # int: 分类任务的类别数量，IMDB,SST-2,HATESPEECH: 2，AGNEWS: 4
    tokenize_max_length: 128          # int: 文本tokenization的最大长度，IMDB: 128, AGNEWS: 256

  attack_config:
    method: RS                        # str: 查询生成方法，可选：RS(随机), TRF(任务相关过滤), DRC(聚类), MeaeQ, AL-RS(主动随机), AL-US(主动不确定性)
    query_num: 320                    # int: 生成查询的数量，IMDB: 320, SST-2: 536, AGNEWS: 960, HATESPEECH: 574
    run_seed_arr: [56]                # list: 随机种子
    pool_data_type: whole             # str: 数据池类型，可选：whole, random_subset, reduced_subset_by_prompt, reduced_subset_by_prompt_integrate
    pool_data_source: imdb            # str: 数据池来源，可选：wiki, imdb, sst2
    pool_subsize: -1                  # int: 数据池子集大小，-1表示使用全部数据，具体值：IMDB/SST-2: 1766, AGNEWS: 212630, HATESPEECH: 1561
    prompt: None                      # str: 用于数据过滤的提示词，IMDB/SST-2: "This is a movie review.", AGNEWS: "This is a news.", HATESPEECH: "This is a hate speech."
    epsilon: -1                       # float: 数据过滤的阈值参数，常设为0.95
    initial_sample_method: random_sentence  # str: 初始采样方法，可选：random_sentence, data_reduction_kmeans, WIKI, RANDOM
    initial_drk_model: None           # str: 数据过滤使用的模型，可选：None, sentence-bert, bart-large-mnli
    al_sample_batch_num: -1           # int: 主动学习采样批次数量，-1表示不使用主动学习
    al_sample_method: None            # str: 主动学习采样方法，可选：random, uncertainty, dr-greedy-select-min-max, dr-greedy-select-max-sum

  train_config:
    batch_size: 32                    # int: 训练批次大小
    optimizer: adam                   # str: 优化器类型
    learning_rate: 3e-5               # float: 学习率
    weight_decay: 1e-4                # float: 权重衰减系数，L2正则化
    num_epochs: 10                    # int: 训练轮数
    weighted_cross_entropy: True      # boolean: 是否使用加权交叉熵损失

  output:
    log_dir: "steal/model_steal/log"  # str: 日志文件输出目录
    model_save_dir: "saved_model"     # str: 模型保存目录
  ```

### 安全风险：

* #### 对抗攻击

  > 主要参考Textattack包：https://github.com/QData/TextAttack/tree/master
  
  配置文件说明：
  
  ```yaml
  attack_args:
    attack: False                   # boolean: 是否开启攻击
    attack_type: AdvAttack          # str: 攻击方法, ‘AdvAttack’代表对抗攻击
    attack_recipe: BAEGarg2019      # str: 具体攻击策略
    use_local_model: True           # boolean: 是否使用本地model
    use_local_tokenizer: True       # boolean: 是否使用本地tokenizer
    use_local_dataset: True         # boolean: 是否使用本地dataset
    model_name_or_path: "LMs/bert_base_uncased"      # str: 攻击所用model在Huggingface的名称或本地路径
    tokenizer_name_or_path: "LMs/bert_base_uncased"  # str: 攻击所用tokenizer在Huggingface的名称或本地路径
    dataset_name_or_path: "data/imdb/test.txt"               # str: 攻击所用dataset在Huggingface的名称或本地路径
    attack_nums: 2                  # int: 攻击次数
    display_full_info: True         # boolean: 是否显示全部过程信息
  ```
  
  支持的具体攻击策略有：
  
  1. (A2TYoo2021) [EMNLP2021] *Yoo J Y, Qi Y.* **Towards Improving Adversarial Training of NLP Models**[A]. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2021: 945-956. [[paper]](https://arxiv.org/abs/2109.00544)
  2. (BAEGarg2019) [EMNLP2020] *Garg S, Ramakrishnan G.* **Bae: BERT-Based Adversarial Examples for Text Classification**[A]. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2020: 6193-6202. [[paper]](https://arxiv.org/abs/2004.01970)
  3. (BERTAttackLi2020) [EMNLP2020] *Li L, Ma R, Guo Q, et al.* **BERT-Attack: Adversarial Attack Against BERT Using BERT**[A]. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2020: 6193-6202. [[paper]](https://arxiv.org/abs/2004.09984)
  4. (GeneticAlgorithmAlzantot2018) [EMNLP2018] *Alzantot M, Sharma Y, Elgohary A, et al.* **Generating Natural Language Adversarial Examples**[A]. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2018: 2890-2896. [[paper]](https://arxiv.org/abs/1804.07998)
  5. (FasterGeneticAlgorithmJia2019) [EMNLP2019] *Jia R, Raghunathan A, Göksel K, et al.* **Certified Robustness to Adversarial Word Substitutions**[A]. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2019: 4127-4140. [[paper]](https://arxiv.org/abs/1909.00986)
  6. (DeepWordBugGao2018) [SPW2018] *Gao J, Lanchantin J, Soffa M L, et al.* **Black-Box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers**[A]. Proceedings of the 2018 IEEE Security and Privacy Workshops (SPW)[C]. Piscataway: IEEE, 2018: 50-56. [[paper]](https://ieeexplore.ieee.org/abstract/document/8424632)
  7. (HotFlipEbrahimi2017) [ACL2018] *Ebrahimi J, Rao A, Lowd D, et al.* **Hotflip: White-Box Adversarial Examples for Text Classification**[A]. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)[C]. 2018: 31-36. [[paper]](https://arxiv.org/abs/1712.06751)
  8. (InputReductionFeng2018) [arXiv2018] *Feng S, Wallace E, Grissom II A, et al.* **Pathologies of Neural Models Make Interpretations Difficult**[J]. arXiv preprint arXiv:1804.07781, 2018. [[paper]](https://arxiv.org/abs/1804.07781)
  9. (Kuleshov2017) [ICLR2018] *Kuleshov V, Thakoor S, Lau T, et al.* **Adversarial Examples for Natural Language Classification Problems**[J]. Proceedings of the International Conference on Learning Representations (ICLR)[C]. Amherst: OpenReview, 2018. [[paper]](https://openreview.net/forum?id=r1QZ3zbAZ)
  10. (MorpheusTan2020) [ACL2020] *Tan S, Joty S, Kan M Y, et al.* **It's Morphin'Time! Combating Linguistic Discrimination with Inflectional Perturbations**[A]. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)[C]. 2020: 2920-2935. [[paper]](https://arxiv.org/abs/2005.04364)
  11. (Seq2SickCheng2018BlackBox) [AAAI2020] *Cheng M, Yi J, Chen P Y, et al.* **Seq2sick: Evaluating the robustness of sequence-to-sequence models with adversarial examples**[A]. Proceedings of the AAAI conference on artificial intelligence[C]. 2020, 34(04): 3601-3608. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5767)
  12. (TextBuggerLi2018) [NDSS2019] *Li J, Ji S, Du T, et al.* **TextBugger: Generating Adversarial Text Against Real-World Applications**[A]. Proceedings of the Network and Distributed System Security Symposium (NDSS)[C]. Virginia: Internet Society, 2019. [[paper]](https://arxiv.org/abs/1812.05271)
  13. (TextFoolerJin2019) [AAAI2020] *Jin D, Jin Z, Zhou J T, et al.* **Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment**[A]. Proceedings of the AAAI conference on artificial intelligence[C]. California: AAAI, 2020, 34(05): 8018-8025. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6311)
  14. (PWWSRen2019) [ACL2019] *Ren S, Deng Y, He K, et al.* **Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency**[A]. Proceedings of the 57th annual meeting of the association for computational linguistics[C]. 2019: 1085-1097. [[paper]](https://aclanthology.org/P19-1103/?amp=1)
  15. (IGAWang2019) [ICLR2020] *Wang X, Jin H, He K.* **Natural language Adversarial Attack and Defense in Word Level**[J]. Proceedings of the International Conference on Learning Representations (ICLR)[C]. Amherst: OpenReview, 2019. [[paper]](https://openreview.net/forum?id=BJl_a2VYPH)
  16. (Pruthi2019) [ACL2019] *Pruthi D, Dhingra B, Lipton Z C.* **Combating Adversarial Misspellings with Robust Word Recognition**[A]. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)[C]. Texas: ACL, 2019: 5582-5591. [[paper]](https://arxiv.org/abs/1905.11268)
  17. (PSOZang2020) [arXiv2019] *Zang Y, Qi F, Yang C, et al.* **Word-Level Textual Adversarial Attacking as Combinatorial Optimization**[J]. arXiv preprint arXiv:1910.12196, 2019. [[paper]](https://arxiv.org/abs/1910.12196)
  18. (CheckList2020) [arXiv2020] *Ribeiro M T, Wu T, Guestrin C, et al.* **Beyond Accuracy: Behavioral Testing of NLP Models with CheckList**[J]. arXiv preprint arXiv:2005.04118, 2020. [[paper]](https://arxiv.org/abs/2005.04118)
  19. (CLARE2020) [arXiv2020] *Li D, Zhang Y, Peng H, et al.* **Contextualized Perturbation for Textual Adversarial Attack**[J]. arXiv preprint arXiv:2009.07502, 2020. [[paper]](https://arxiv.org/abs/2009.07502)
  
* #### 后门攻击

  > 主要参考Openbackdoor包：https://github.com/thunlp/OpenBackdoor?tab=readme-ov-file#usage
  
  配置文件说明：
  
  ```yaml
  attack_args:
    attack: True                    # boolean: 是否开启攻击
    attack_type: BackdoorAttack     # str: 攻击方法, ‘BackdoorAttack’代表后门攻击
    use_local_model: True           # boolean: 是否使用本地model
    model: "bert"                   # str: 攻击目标model的名称
    model_name_or_path: "LMs/bert_base_uncased"       # str: 攻击目标model在Huggingface的名称或本地路径
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
    sample_metrics: []              # list: 评估分数, ['ppl', 'use', 'grammar']
  ```
  
  支持的具体攻击策略有：
  
  1. (BadNets) [arXiv2017] *Gu T, Dolan-Gavitt B, Garg S.* **Badnets: Identifying vulnerabilities in the machine learning model supply chain**[J]. arXiv preprint arXiv:1708.06733, 2017. [[paper]](https://arxiv.org/abs/1708.06733)
  2. (AddSent) [ACCESS2019] *Dai J, Chen C, Li Y.* **A backdoor attack against lstm-based text classification systems**[J]. IEEE Access, 2019, 7: 138872-138878. [[paper]](https://arxiv.org/pdf/1905.12457.pdf)
  3. (SynBkd) [ACL/IJCNLP2021] *Qi F, Li M, Chen Y, et al.* **Hidden killer: Invisible textual backdoor attacks with syntactic trigger**[J]. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing. 2021: 443-453. [[paper]](https://arxiv.org/pdf/2105.12400.pdf)
  4. (StyleBkd) [arXiv2021] *Qi F, Chen Y, Zhang X, et al.* **Mind the style of text! adversarial and backdoor attacks based on text style transfer**[J]. arXiv preprint arXiv:2110.07139, 2021. [[paper]](https://arxiv.org/pdf/2110.07139.pdf)
  5. (POR) [CCS2021] *Shen L, Ji S, Zhang X, et al.* **Backdoor pre-trained models can transfer to all**[J]. arXiv preprint arXiv:2111.00197, 2021. [[paper]](https://arxiv.org/abs/2111.00197)
  6. (TrojanLM) [EuroS&P2021] *Zhang X, Zhang Z, Ji S, et al.* **Trojaning language models for fun and profit**[A]. 2021 IEEE European Symposium on Security and Privacy (EuroS&P)[A]. IEEE, 2021: 179-197. [[paper]](https://arxiv.org/abs/2008.00312)
  7. (SOS) [ACL/IJCNLP2021] *Yang W, Lin Y, Li P, et al.* **Rethinking stealthiness of backdoor attack against nlp models**[C]//Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 2021: 5543-5557. [[paper]](https://aclanthology.org/2021.acl-long.431)
  8. (LWP) [EMNLP2021] *Li L, Song D, Li X, et al.* **Backdoor attacks on pre-trained models by layerwise weight poisoning**[J]. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2021: 3023-3032. [[paper]](https://aclanthology.org/2021.emnlp-main.241.pdf)
  9. (EP) [NAACL-HLT2021] *Yang W, Li L, Zhang Z, et al.* **Be careful about poisoned word embeddings: Exploring the vulnerability of the embedding layers in NLP models**[J]. NAACL-HLT. 2021: 2048-2058. [[paper]](https://aclanthology.org/2021.naacl-main.165)
  10. (NeuBA) [MathIntellRes2023] *Zhang Z, Xiao G, Li Y, et al.* **Red alarm for pre-trained models: Universal vulnerability to neuron-level backdoor attacks**[J]. Machine Intelligence Research, 2023, 20(2): 180-193. [[paper]](https://arxiv.org/abs/2101.06969)
  11. (LWS) [ACL/IJCNLP2021] *Qi F, Yao Y, Xu S, et al.* **Turn the combination lock: Learnable textual backdoor attacks via word substitution**[J]. ACL/IJCANLP. 2021: 4873-4883. [[paper]](https://aclanthology.org/2021.acl-long.377.pdf)
  12. (RIPPLES) [EMNLP2020] *Kurita K, Michel P, Neubig G.* **Weight poisoning attacks on pre-trained models**[J].Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2020: 3023-3032. [[paper]](https://aclanthology.org/2020.acl-main.249.pdf)
  
  支持的具体防御策略有：
  
  1. (BKI) [Neurcomputing2021] [Neurocomputing2021] *Chen C, Dai J.* **Mitigating backdoor attacks in lstm-based text classification systems by backdoor keyword identification**[J]. Neurocomputing, 2021, 452: 253-262. [[paper]](https://arxiv.org/ans/2007.12070)
  2. (ONION) [EMNLP2021] *Qi F, Chen Y, Li M, et al.* **ONION: A Simple and Effective Defense Against Textual Backdoor Attacks**[A]. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2021: 9558-9566. [[paper]](https://arxiv.org/abs/2011.10369)
  3. (STRIP) [TDSC2022] *Gao Y, Kim Y, Doan B G, et al.* **Design and evaluation of a multi-domain trojan detection method on deep neural networks**[J]. IEEE Transactions on Dependable and Secure Computing, 2021, 19(4): 2349-2364. [[paper]](https://arxiv.org/abs/1911.10312)
  4. (RAP) [EMNLP2021] *Yang W, Lin Y, Li P, et al.* **Rap: Robustness-aware perturbations for defending against backdoor attacks on nlp models**[A]. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)[C]. Texas: ACL, 2021: 8365-8381. [[paper]](https://arxiv.org/abs/2110.07831)
  5. (CUBE) [NeurIPS2022] *Cui G, Yuan L, He B, et al.* **A unified evaluation of textual backdoor learning: Frameworks and benchmarks**[A]. Advances in Neural Information Processing Systems, 2022, 35: 5009-5023. [[paper]](https://arxiv.org/abs/2206.08514)
  
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

