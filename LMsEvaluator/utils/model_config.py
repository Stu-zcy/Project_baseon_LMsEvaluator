import os
import torch
import logging
from model.BasicBert import BertConfig
from torch.utils.tensorboard import SummaryWriter
from utils.log_helper import logger_init


class ModelConfig:
    def __init__(self, dataset_dir, model_dir, dataset_type='.txt', use_gpu=True, config_parser=None):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'datasets', dataset_dir)
        self.pretrained_model_dir = os.path.join(self.project_dir, "LMs", model_dir)
        self.vocab_path = os.path.join(str(self.pretrained_model_dir), 'vocab.txt')
        self.device = torch.device('mps' if (use_gpu and torch.backends.mps.is_available()) else
                                   'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
        self.train_file_path = os.path.join(str(self.dataset_dir), 'train' + dataset_type)
        self.val_file_path = os.path.join(str(self.dataset_dir), 'val' + dataset_type)
        self.test_file_path = os.path.join(str(self.dataset_dir), 'test' + dataset_type)
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, 'model.pt')
        self.config_parser = config_parser
        self.log_level = logging.INFO

        self.vocab_size = 21128,
        self.hidden_size = 768,
        self.num_hidden_layers = 12,
        self.num_attention_heads = 12,
        self.intermediate_size = 3072,
        self.pad_token_id = 0,
        self.hidden_act = "gelu",
        self.hidden_dropout_prob = 0.1,
        self.attention_probs_dropout_prob = 0.1,
        self.max_position_embeddings = 512,
        self.type_vocab_size = 2,
        self.initializer_range = 0.02

    def log_init(self, log_file_name='single'):
        logger_init(log_file_name=log_file_name, log_level=self.log_level,
                    log_dir=self.logs_save_dir, only_file=False)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def bert_import(self):
        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(str(self.pretrained_model_dir), "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value

    def log_config(self):
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


class ModelTaskForChineseNER(ModelConfig):
    def __init__(self, dataset_dir, model_dir, dataset_type='.txt', use_gpu=True, split_sep=' ',
                 config_parser=None,username='default'):
        super().__init__(dataset_dir, model_dir, dataset_type, use_gpu, config_parser)
        self.model_save_name = "ner_model.pt"
        self.writer = SummaryWriter("runs")
        self.split_sep = split_sep
        self.is_sample_shuffle = True
        self.batch_size = 12
        self.max_sen_len = None
        self.epochs = config_parser['task_config']['epochs'] if (config_parser is not None) else 10
        self.learning_rate = 1e-5
        self.model_val_per_epoch = 2
        self.entities = {'O': 0, 'B-ORG': 1, 'B-LOC': 2, 'B-PER': 3, 'I-ORG': 4, 'I-LOC': 5, 'I-PER': 6}
        self.num_labels = len(self.entities)
        self.ignore_idx = -100
        self.log_level = logging.DEBUG
        self.log_init(log_file_name=username+'_ner')
        self.bert_import()
        self.log_config()


class ModelTaskForMultipleChoice(ModelConfig):
    def __init__(self, dataset_dir, model_dir, dataset_type='.csv', use_gpu=True, config_parser=None,username='default'):
        super().__init__(dataset_dir, model_dir, dataset_type, use_gpu, config_parser)
        self.is_sample_shuffle = True
        self.batch_size = 16
        self.max_sen_len = None
        self.num_labels = 4  # num_choice
        self.learning_rate = 2e-5
        self.epochs = config_parser['task_config']['epochs'] if (config_parser is not None) else 10
        self.model_val_per_epoch = 2
        self.log_level = logging.INFO
        self.log_init(log_file_name=username+'_choice')
        self.bert_import()
        self.log_config()


class ModelTaskForPairSentenceClassification(ModelConfig):
    def __init__(self, dataset_dir, model_dir, dataset_type='.txt', use_gpu=True, split_sep='_!_',
                 config_parser=None,username='default'):
        super().__init__(dataset_dir, model_dir, dataset_type, use_gpu, config_parser)
        self.split_sep = split_sep
        self.is_sample_shuffle = True
        self.batch_size = 8
        self.learning_rate = 3.5e-5
        self.max_sen_len = None
        self.num_labels = 3
        self.epochs = config_parser['task_config']['epochs'] if (config_parser is not None) else 2
        self.model_val_per_epoch = 2
        self.log_level = logging.INFO
        self.log_init(log_file_name=username+'_pair')
        self.bert_import()
        self.log_config()


class ModelTaskForPretraining(ModelConfig):
    def __init__(self, dataset_dir, model_dir, dataset_type='.txt', use_gpu=True,
                 data_name="songci", config_parser=None,username='default'):
        super().__init__(dataset_dir, model_dir, dataset_type, use_gpu, config_parser)
        self.data_name = data_name
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{data_name}.bin')
        self.writer = SummaryWriter(f"runs/{data_name}")
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 16
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2022
        self.learning_rate = 4e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = config_parser['task_config']['epochs'] if (config_parser is not None) else 200
        self.model_val_per_epoch = 1
        self.log_level = logging.DEBUG
        self.log_init(log_file_name=username+'_'+data_name)
        self.bert_import()
        self.log_config()


class ModelTaskForSingleSentenceClassification(ModelConfig):
    def __init__(self, dataset_dir, model_dir, dataset_type='.txt', use_gpu=True, split_sep='_!_',
                 config_parser=None,username='default'):
        # super(ModelTest, self).__init__(dataset_dir, model_dir, dataset, use_gpu, config_parser)
        super().__init__(dataset_dir, model_dir, dataset_type, use_gpu, config_parser)
        self.split_sep = split_sep
        self.is_sample_shuffle = True
        self.batch_size = 1
        self.max_sen_len = None
        self.num_labels = 15
        self.epochs = config_parser['task_config']['epochs'] if (config_parser is not None) else 10
        self.model_val_per_epoch = 2
        self.log_level = logging.INFO
        self.log_init(log_file_name=username+'_single')
        self.bert_import()
        self.log_config()


class ModelTaskForSQuADQuestionAnswering(ModelConfig):
    def __init__(self, dataset_dir, model_dir, dataset_type='.json', use_gpu=True, config_parser=None,username='default'):
        # super(ModelTest, self).__init__(dataset_dir, model_dir, dataset, use_gpu)
        super().__init__(dataset_dir, model_dir, dataset_type, use_gpu, config_parser)
        self.train_file_path = os.path.join(str(self.dataset_dir), "train" + dataset_type)
        self.test_file_path = os.path.join(str(self.dataset_dir), "test" + dataset_type)
        self.n_best_size = 10  # 对预测出的答案近后处理时，选取的候选答案数量
        self.max_answer_len = 30  # 在对候选进行筛选时，对答案最大长度的限制
        self.is_sample_shuffle = True  # 是否对训练集进行打乱
        self.use_torch_multi_head = False  # 是否使用PyTorch中的multihead实现
        self.batch_size = 12
        self.max_sen_len = 384  # 最大句子长度，即 [cls] + question ids + [sep] +  context ids + [sep] 的长度
        self.max_query_len = 64  # 表示问题的最大长度，超过长度截取
        self.learning_rate = 3.5e-5
        self.doc_stride = 128  # 滑动窗口一次滑动的长度
        self.epochs = config_parser['task_config']['epochs'] if (config_parser is not None) else 200
        self.model_val_per_epoch = 1
        self.log_level = logging.DEBUG
        self.log_init(log_file_name='qa')
        self.bert_import()
        self.log_config()


if __name__ == "__main__":
    # 下游任务模型配置示例
    # model_config = ModelConfig("SingleSentenceClassification", "bert_base_chinese", ".txt", True)
    # model_config = ModelTaskForSingleSentenceClassification("SingleSentenceClassification", "bert_base_chinese",
    #                                                         ".txt", True, '_!_')
    # model_config = ModelTaskForSQuADQuestionAnswering("SQuAD", "bert_base_uncased_english", "v1-", ".json", True)
    # model_config = ModelTaskForChineseNER("ChineseNER", "bert_base_chinese", ".txt", True, ' ')
    # model_config = ModelTaskForMultipleChoice("MultipleChoice", "bert_base_uncased_english", ".csv", True)
    # model_config = ModelTaskForPairSentenceClassification("PairSentenceClassification", "bert_base_uncased_english",
    #                                                       ".txt", True, '_!_')
    # model_config = ModelTaskForPretraining("SongCi", "bert_base_chinese", ".txt", True, "songci")
    model_config = ModelTaskForPretraining("WikiText", "bert_base_uncased_english", ".tokens", True, "wiki2")
