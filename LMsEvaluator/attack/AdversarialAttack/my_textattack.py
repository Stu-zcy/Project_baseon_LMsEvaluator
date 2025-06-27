import logging
import os
import csv
import transformers
from utils.my_prettytable import MyPrettyTable
from attack.base_attack import BaseAttack
from utils.my_exception import print_red


# bert_base_uncased_english + Huggingface("imdb")
# +-------------------------------+--------+
# | Attack Results                |        |
# +-------------------------------+--------+
# | Number of successful attacks: | 18     |
# | Number of failed attacks:     | 0      |
# | Number of skipped attacks:    | 2      |
# | Original accuracy:            | 90.0%  |
# | Accuracy under attack:        | 0.0%   |
# | Attack success rate:          | 100.0% |
# | Average perturbed word %:     | 11.04% |
# | Average num. words per input: | 203.75 |
# | Avg num queries:              | 624.67 |
# +-------------------------------+--------+

class MyTextAttack(BaseAttack):
    def __init__(self, config_parser, attack_config, use_local_model=False, use_local_tokenizer=False,
                 use_local_dataset=False, model_name_or_path=None, tokenizer_name_or_path=None,
                 dataset_name_or_path=None, display_full_info=False):
        super().__init__(config_parser, attack_config)
        self.use_local_model = use_local_model
        self.use_local_tokenizer = use_local_tokenizer
        self.use_local_dataset = use_local_dataset
        self.display_full_info = display_full_info
        self.my_handlers = logging.getLogger().handlers

        # 项目路径获取 + 检查
        self.project_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])
        self.model_name_or_path = self.__get_name_or_path(self.use_local_model,
                                                          model_name_or_path, "model")
        self.tokenizer_name_or_path = self.__get_name_or_path(self.use_local_tokenizer,
                                                              tokenizer_name_or_path, "tokenizer")
        self.dataset_name_or_path = self.__get_name_or_path(self.use_local_dataset,
                                                            dataset_name_or_path, "dataset")

    def attack(self):
        import textattack
        from textattack import attack_recipes
        model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        self.dataset = self.get_local_dataset()

        attack_recipe = self.attack_config['attack_recipe']

        if attack_recipe == 'TextFoolerJin2019':
            attack_model = attack_recipes.TextFoolerJin2019.build(self.model_wrapper)
        elif attack_recipe == 'BERTAttackLi2020':
            attack_model = attack_recipes.BERTAttackLi2020.build(self.model_wrapper)
        elif attack_recipe == 'A2TYoo2021':
            attack_model = attack_recipes.A2TYoo2021.build(self.model_wrapper)
        elif attack_recipe == 'BAEGarg2019':
            attack_model = attack_recipes.BAEGarg2019.build(self.model_wrapper)
        elif attack_recipe == 'GeneticAlgorithmAlzantot2018':
            attack_model = attack_recipes.GeneticAlgorithmAlzantot2018.build(self.model_wrapper)
        elif attack_recipe == 'FasterGeneticAlgorithmJia2019':
            attack_model = attack_recipes.FasterGeneticAlgorithmJia2019.build(self.model_wrapper)
        elif attack_recipe == 'DeepWordBugGao2018':
            attack_model = attack_recipes.DeepWordBugGao2018.build(self.model_wrapper)
        elif attack_recipe == 'HotFlipEbrahimi2017':
            attack_model = attack_recipes.HotFlipEbrahimi2017.build(self.model_wrapper)
        elif attack_recipe == 'InputReductionFeng2018':
            attack_model = attack_recipes.InputReductionFeng2018.build(self.model_wrapper)
        elif attack_recipe == 'Kuleshov2017':
            attack_model = attack_recipes.Kuleshov2017.build(self.model_wrapper)
        elif attack_recipe == 'MorpheusTan2020':
            attack_model = attack_recipes.MorpheusTan2020.build(self.model_wrapper)
        elif attack_recipe == 'Seq2SickCheng2018BlackBox':
            attack_model = attack_recipes.Seq2SickCheng2018BlackBox.build(self.model_wrapper)
        elif attack_recipe == 'TextBuggerLi2018':
            attack_model = attack_recipes.TextBuggerLi2018.build(self.model_wrapper)
        elif attack_recipe == 'PWWSRen2019':
            attack_model = attack_recipes.PWWSRen2019.build(self.model_wrapper)
        elif attack_recipe == 'IGAWang2019':
            attack_model = attack_recipes.IGAWang2019.build(self.model_wrapper)
        elif attack_recipe == 'Pruthi2019':
            attack_model = attack_recipes.Pruthi2019.build(self.model_wrapper)
        elif attack_recipe == 'PSOZang2020':
            attack_model = attack_recipes.PSOZang2020.build(self.model_wrapper)
        elif attack_recipe == 'CheckList2020':
            attack_model = attack_recipes.CheckList2020.build(self.model_wrapper)
        elif attack_recipe == 'CLARE2020':
            attack_model = attack_recipes.CLARE2020.build(self.model_wrapper)
        elif attack_recipe == 'FrenchRecipe':
            attack_model = attack_recipes.FrenchRecipe.build(self.model_wrapper)
        elif attack_recipe == 'SpanishRecipe':
            attack_model = attack_recipes.SpanishRecipe.build(self.model_wrapper)
        elif attack_recipe == 'ChineseRecipe':
            attack_model = attack_recipes.ChineseRecipe.build(self.model_wrapper)
        else:
            print_red("ERROR: UNKNOWN ATTACK RECIPE.")
            print_red("Please check the attackArgs.attackRecipe config in config.yaml.")
            raise SystemError

        attack_args = textattack.AttackArgs(
            num_examples=self.attack_config['attack_nums'],
            log_to_csv=os.path.join(self.project_path, "attack/AdversarialAttack/log.csv"),
            checkpoint_interval=5,
            checkpoint_dir="checkpoints",
            disable_stdout=True,
        )
        attacker = textattack.Attacker(attack_model, self.dataset, attack_args)

        if self.display_full_info:
            logging.getLogger().handlers = self.my_handlers

        attacker.attack_dataset()
        self.__result_show()

        # TODO 对抗训练算法实现
        # train_dataset = self.dataset
        # eval_dataset = self.dataset
        # # train_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="train")
        # # eval_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
        # training_args = textattack.TrainingArgs(
        #     num_epochs=3,
        #     num_clean_epochs=1,
        #     num_train_adv_examples=1000,
        #     learning_rate=5e-5,
        #     per_device_train_batch_size=8,
        #     gradient_accumulation_steps=4,
        #     log_to_tb=False,
        # )

        # model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        # tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        # model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

        # model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        # tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        # self.model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        #
        # attack_model = attack_recipes.TextFoolerJin2019.build(self.model_wrapper)
        #
        # trainer = textattack.Trainer(
        #     self.model_wrapper,
        #     "classification",
        #     attack_model,
        #     train_dataset,
        #     eval_dataset,
        #     training_args,
        # )
        # trainer.train()

    def __result_show(self):
        with open('attack/AdversarialAttack/log.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            total_num, successful_num, skipped_num, failed_num = 0, 0, 0, 0
            for row in csv_reader:
                if row[-1] == 'Successful':
                    successful_num += 1
                elif row[-1] == 'Skipped':
                    skipped_num += 1
                elif row[-1] == 'Failed':
                    failed_num += 1

            total_num = successful_num + skipped_num + failed_num
            if total_num == 0:
                ori_acc, acc_under_attack = 0, 0
            else:
                ori_acc = (successful_num + failed_num) / total_num * 100
                acc_under_attack = failed_num / total_num * 100
            if successful_num + failed_num == 0:
                attack_acc = 0
            else:
                attack_acc = successful_num / (successful_num + failed_num) * 100

            table_show(successful_num=successful_num, failed_num=failed_num, skipped_num=skipped_num, ori_acc=ori_acc,
                       acc_under_attack=acc_under_attack, attack_acc=attack_acc)

    def __get_name_or_path(self, boolean, name_or_path, info):
        if boolean:
            result = os.path.join(self.project_path, name_or_path)
            try:
                if not os.path.exists(result):
                    raise FileNotFoundError(
                        print_red("FileNotFoundError: No such file or directory: " + result +
                                  ". Please check the path to local " + info + " in my_textattack.py."))
            except Exception as e:
                print(e)
                raise SystemError
        else:
            result = os.path.join("textattack", name_or_path)
        return result

    def get_local_dataset(self):
        """
        :return: dataset
        P.S.
        local_dataset: [(input, output)]
        e.g. local_dataset = [('Today was ...', 1), ('This movie is ...', 0)]
        """
        from textattack import datasets
        if self.use_local_dataset:
            local_dataset = []
            with open(self.dataset_name_or_path, 'r', encoding='utf-8') as file:
                data_lines = file.readlines()
                for data_line in data_lines:
                    data_line_split = data_line.split("_!_")
                    temp = ""
                    for index in range(len(data_line_split) - 1):
                        temp = temp + data_line_split[index]
                    local_dataset.append((temp, int(data_line_split[-1])))
            return datasets.Dataset(local_dataset)
        else:
            return datasets.HuggingFaceDataset(self.dataset_name_or_path, split="test")


def dataset_getter(path):
    from textattack import datasets
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(
                print_red("FileNotFoundError: No such file or directory: " + path +
                          ". Please check the path to local dataset in my_textattack.py."))
    except Exception as e:
        print(e)
        raise SystemError

    local_dataset = []
    with open(path, 'r', encoding='utf-8') as file:
        data_lines = file.readlines()
        for data_line in data_lines:
            data_line_split = data_line.split("_!_")
            temp = ""
            for index in range(len(data_line_split) - 1):
                temp = temp + data_line_split[index]
            local_dataset.append((temp, int(data_line_split[-1])))
    return datasets.Dataset(local_dataset)


def table_show(successful_num, failed_num, skipped_num, ori_acc, acc_under_attack, attack_acc):
    table = MyPrettyTable()
    table.add_field_names(['Attack Results', ''])
    table.add_row(['Number of successful attacks:', successful_num])
    table.add_row(['Number of failed attacks:', failed_num])
    table.add_row(['Number of skipped attacks:', skipped_num])
    table.add_row(['Original accuracy:', f"{ori_acc:.1f}%"])
    table.add_row(['Accuracy under attack:', f"{acc_under_attack:.1f}%"])
    table.add_row(['Attack success rate:', f"{attack_acc:.1f}%"])
    table.set_align('Attack Results', 'l')
    table.set_align('', 'l')
    table.print_table()
    table.logging_table()


# 本地测试
if __name__ == "__main__":
    """
    AdvAttack模块功能测试
    """
    import textattack

    # 项目路径获取
    projectPath = os.path.dirname(os.path.abspath(__file__))
    projectPath = "/".join(projectPath.split("/")[:-2])

    model = transformers.AutoModelForSequenceClassification.from_pretrained("../../LMs/bert_base_uncased_english")
    tokenizer = transformers.AutoTokenizer.from_pretrained("../../LMs/bert_base_uncased_english")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    datasetPath = os.path.join(projectPath, "datasets", "imdb/test.txt")
    localDataset = dataset_getter(datasetPath)
    # dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
        num_examples=20,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=True
    )
    attacker = textattack.Attacker(attack, localDataset, attack_args)
    attacker.attack_dataset()

    with open('log.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        total_num, successful_num, skipped_num, failed_num = 0, 0, 0, 0
        for row in csv_reader:
            if row[-1] == 'Successful':
                successful_num += 1
            elif row[-1] == 'Skipped':
                skipped_num += 1
            elif row[-1] == 'Failed':
                failed_num += 1

        total_num = successful_num + skipped_num + failed_num
        if total_num == 0:
            ori_acc, acc_under_attack = 0, 0
        else:
            ori_acc = (successful_num + failed_num) / total_num * 100
            acc_under_attack = failed_num / total_num * 100
        if successful_num + failed_num == 0:
            attack_acc = 0
        else:
            attack_acc = successful_num / (successful_num + failed_num) * 100

        table_show(successful_num, failed_num, skipped_num, ori_acc, acc_under_attack, attack_acc)
