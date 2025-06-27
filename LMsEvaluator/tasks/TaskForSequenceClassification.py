import os

from datasets import load_metric, load_dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TaskForSequenceClassification:
    def __init__(self, dataset_dir='imdb', model_dir=None):
        # pass
        dataset_loader = load_dataset(os.path.join(project_path, 'datasets', dataset_dir))
        print(dataset_loader)

        # model = AutoModelForSequenceClassification.from_pretrained(
        #     '/Users/zkzhu/Project/PycharmProjects/LMsEvaluator/LMs/bert_base_uncased_english')
        #
        # metric = load_metric("accuracy")
        # training_args = TrainingArguments(evaluation_strategy="epoch")
        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #
        # )

    def train(self):
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
        self.attack()


if __name__ == "__main__":
    print(project_path)
    TaskForSequenceClassification()
