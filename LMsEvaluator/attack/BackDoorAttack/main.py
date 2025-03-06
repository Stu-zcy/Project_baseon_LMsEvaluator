import os
import logging
import numpy as np
from utils.my_exception import print_red
from attack.base_attack import BaseAttack
from utils.my_prettytable import MyPrettyTable


class MyBackDoorAttack(BaseAttack):
    def __init__(self, config_parser, attack_config, use_local_model=False,
                 model="bert", model_name_or_path=None, poison_dataset=None, target_dataset=None,
                 poisoner=None, train=None, defender=None, display_full_info=False, sample_metrics=[]):
        super().__init__(config_parser, attack_config)
        self.model = model
        self.use_local_model = use_local_model
        self.poison_dataset = poison_dataset
        self.target_dataset = target_dataset
        self.poisoner = poisoner
        self.train = train
        self.defender = defender
        self.display_full_info = display_full_info
        self.my_handlers = logging.getLogger().handlers
        self.sample_metrics = sample_metrics

        # 项目路径获取 + 检查
        self.project_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])
        self.model_name_or_path = self.__get_name_or_path(self.use_local_model,
                                                          model_name_or_path, "model")

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
            result = name_or_path
        return result

    def attack(self):
        import openbackdoor as ob
        from openbackdoor import load_dataset
        from openbackdoor.defenders import BKIDefender, CUBEDefender, ONIONDefender, RAPDefender, STRIPDefender
        # choose BERT as victim model
        victim = ob.PLMVictim(model=self.model, path=self.model_name_or_path)
        # choose BadNet attacker
        attacker = ob.Attacker(poisoner=self.poisoner, train=self.train, sample_metrics=self.sample_metrics)
        # attacker = ob.Attacker(poisoner=self.poisoner, train=self.train, sample_metrics=['grammar'])
        # choose SST-2 as the poison data
        poison_dataset = load_dataset(name=self.poison_dataset)

        if self.display_full_info:
            logging.getLogger().handlers = self.my_handlers

        defender = None
        if self.defender == "onion":
            defender = ONIONDefender()
        elif self.defender == "bki":
            defender = BKIDefender()
        elif self.defender == "cube":
            defender = CUBEDefender()
        elif self.defender == "rap":
            defender = RAPDefender()
        elif self.defender == "strip":
            defender = STRIPDefender()

        # launch attack
        victim = attacker.attack(victim, poison_dataset, defender)
        # choose SST-2 as the target data
        target_dataset = load_dataset(name=self.target_dataset)
        # evaluate attack results
        result = attacker.eval(victim, target_dataset, defender)
        print(result)
        table_ppl, table_use, table_grammar = "nan", "nan", "nan"
        if result['ppl'] is not np.nan:
            table_ppl = "{:.3f}".format(result['ppl'])
        if result['use'] is not np.nan:
            table_use = "{:.3f}".format(result['use'])
        if result['grammar'] is not np.nan:
            table_grammar = "{:.3f}".format(result['grammar'])

        table = MyPrettyTable()
        table.add_field_names(['BackdoorAttack Attack Results', ''])
        table.add_row(['Poison Dataset:', self.poison_dataset])
        table.add_row(['Poisoner:', self.poisoner['name']])
        table.add_row(['Test Clean Accuracy:', f"{result['test-clean']['accuracy']:.3f}%"])
        table.add_row(['Test Poison Accuracy:', f"{result['test-poison']['accuracy']:.3f}%"])
        table.add_row(['PPL:', table_ppl])
        table.add_row(['USE:', table_use])
        table.add_row(['GRAMMAR:', table_grammar])
        table.set_align('BackdoorAttack Attack Results', 'l')
        table.set_align('', 'l')
        table.print_table()
        table.logging_table()


# 本地测试
if __name__ == '__main__':
    """
    BackdoorAttack模块功能测试
    """
    import openbackdoor as ob
    from openbackdoor import load_dataset

    victim = ob.PLMVictim(model="bert", path="../../LMs/bert_base_uncased_english")
    attacker = ob.Attacker(poisoner={"name": "badnets"}, train={"name": "base", "batch_size": 32})
    poison_dataset = load_dataset(name="sst-2")

    victim = attacker.attack(victim, poison_dataset)
    target_dataset = load_dataset(name="sst-2")
    attacker.eval(victim, target_dataset)
