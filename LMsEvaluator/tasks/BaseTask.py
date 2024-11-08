import sys
import utils.model_config as model_config

sys.path.append('..')


class BaseTask:
    """
    下游任务模版类
    """

    def __init__(self, dataset_dir, model_dir, use_gpu=True, config_parser=None):
        self.config = model_config.ModelConfig(dataset_dir, model_dir, use_gpu, config_parser)
        # self.poisoning_label = ["0", "1"]
        # if dataset_dir == "GLUE/mnli" or dataset_dir == "PairSentenceClassification":
        #     self.poisoning_label = ["0", "1", "2"]
        # elif dataset_dir == "MultipleChoice":
        #     self.poisoning_label = ["0", "1", "2", "3"]
        # elif dataset_dir == "SingleSentenceClassification":
        #     self.poisoning_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
        # elif dataset_dir == "ChineseNER":
        #     self.poisoning_label = ["0", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]

    def train(self, attack=False):
        pass

    def inference(self):
        pass

    def evaluate(self):
        pass

    def attack(self):
        pass

    def run(self):
        self.train()
        self.inference()


if __name__ == "__main__":
    BaseTask("dataset_dir", "model_dir", True)
