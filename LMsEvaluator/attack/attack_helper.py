import logging

from attack.SWAT.main import SWAT
from utils.my_exception import print_red
from attack.base_attack import BaseAttack
from attack.GIAforNLP.my_GIA_for_NLP import MyGIAforNLP
from attack.AdvAttack.my_textattack import MyTextAttack
from attack.BackDoorAttack.main import MyBackDoorAttack
from attack.PoisoningAttack.main import PoisoningAttack


class AttackHelper:
    def __init__(self, attack_type, config_parser, attack_config, **kwargs):
        self.attack_type = attack_type
        self.config_parser = config_parser
        self.attack_config = attack_config
        self.attack_model = BaseAttack(self.config_parser, self.attack_config)
        self.__config_check()
        if self.attack_type == "GIAforNLP":
            logging.info("Checking the config of GIAforNLP.")
            self.attack_model = MyGIAforNLP(
                attack_config=self.attack_config,
                model=kwargs['model'],
                tokenizer=kwargs['tokenizer'],
                train_iter=kwargs['train_iter'],
                distance_func=kwargs['distance_func'],
                gradients=kwargs['gradients'],
                config_parser=self.config_parser,
                data_loader=kwargs['data_loader'],
            )
        elif self.attack_type == "AdvAttack":
            logging.info("Checking the config of AdvAttack.")
            self.attack_model = MyTextAttack(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                use_local_model=self.attack_config['use_local_model'],
                use_local_tokenizer=self.attack_config['use_local_tokenizer'],
                use_local_dataset=self.attack_config['use_local_dataset'],
                model_name_or_path=self.attack_config['model_name_or_path'],
                tokenizer_name_or_path=self.attack_config['tokenizer_name_or_path'],
                dataset_name_or_path=self.attack_config['dataset_name_or_path'],
                display_full_info=self.attack_config['display_full_info'],
            )
        elif self.attack_type == "SWAT":
            logging.info("Checking the config of SWAT.")
            self.attack_model = SWAT(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                use_local_model=self.attack_config['use_local_model'],
                use_local_tokenizer=self.attack_config['use_local_tokenizer'],
                model_name_or_path=self.attack_config['model_name_or_path'],
                tokenizer_name_or_path=self.attack_config['tokenizer_name_or_path'],
                dataset_name_or_path=self.attack_config['dataset_name_or_path'],
                display_full_info=self.attack_config['display_full_info'],
            )
        elif self.attack_type == "BackDoorAttack":
            logging.info("Checking the config of BackDoorAttack.")
            self.attack_model = MyBackDoorAttack(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                model=self.attack_config['model'],
                use_local_model=self.attack_config['use_local_model'],
                model_name_or_path=self.attack_config['model_name_or_path'],
                poison_dataset=self.attack_config['poison_dataset'],
                target_dataset=self.attack_config['target_dataset'],
                poisoner=self.attack_config['poisoner'],
                train=self.attack_config['train'],
                defender=self.attack_config['defender'],
                display_full_info=self.attack_config['display_full_info'],
            )
        elif self.attack_type == "PoisoningAttack":
            logging.info("Checking the config of PoisoningAttack")
            self.attack_model = PoisoningAttack(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                poisoning_rate=self.attack_config['poisoning_rate'],
                epochs=self.attack_config['epochs'],
                task_config=kwargs['task_config'],
                model=kwargs['model'],
                optimizer=kwargs['optimizer'],
                bert_tokenize=kwargs['bert_tokenize'],
                display_full_info=self.attack_config['display_full_info'],
            )

    def attack(self):
        return self.attack_model.attack()

    def __config_check(self):
        GIAforNLP_config = [
            'optimizer',
            'attack_batch',
            'attack_nums',
            'distance_func',
            'attack_lr',
            'attack_iters',
        ]
        AdvAttack_config = [
            'attack_recipe',
            'use_local_model',
            'use_local_tokenizer',
            'use_local_dataset',
            'model_name_or_path',
            'tokenizer_name_or_path',
            'dataset_name_or_path',
            'attack_nums',
            'display_full_info',
        ]
        SWAT_config = [
            'use_local_model',
            'use_local_tokenizer',
            'model_name_or_path',
            'tokenizer_name_or_path',
            'dataset_name_or_path',
            'display_full_info',
        ]
        BackDoorAttack_config = [
            'use_local_model',
            'model',
            'model_name_or_path',
            'poison_dataset',
            'target_dataset',
            'poisoner',
            'train',
            'defender',
            'display_full_info',
        ]
        PoisoningAttack_config = [
            'poisoning_rate',
            'epochs',
            'display_full_info',
        ]
        temp_config = []
        if self.attack_type == "GIAforNLP":
            temp_config = GIAforNLP_config
        elif self.attack_type == "AdvAttack":
            temp_config = AdvAttack_config
        elif self.attack_type == "SWAT":
            temp_config = SWAT_config
        elif self.attack_type == "BackDoorAttack":
            temp_config = BackDoorAttack_config
        elif self.attack_type == "PoisoningAttack":
            temp_config = PoisoningAttack_config

        for config in temp_config:
            if config not in self.attack_config:
                print_red("NoAttackConfigFound: No attack_args." + config + " in the config.yaml.")
                raise SystemError
