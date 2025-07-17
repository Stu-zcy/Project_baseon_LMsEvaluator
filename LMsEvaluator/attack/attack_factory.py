import logging

from attack.NOP.main import NOP
from attack.FET.main import FET
from attack.RLMI.main import RLMI
from utils.my_exception import print_red
from attack.base_attack import BaseAttack
from attack.GIAforNLP.my_GIA_for_NLP import MyGIAforNLP
from attack.AdversarialAttack.my_textattack import MyTextAttack
from attack.BackdoorAttack.main import MyBackDoorAttack
from attack.PoisoningAttack.main import PoisoningAttack
from attack.MeaeQ.my_meaeq import MyMeaeQ

class AttackFactory:
    def __init__(self, attack_type, config_parser, attack_config, device, **kwargs):
        self.attack_type = attack_type
        self.config_parser = config_parser
        self.attack_config = attack_config
        self.device = device
        self.attack_mode = BaseAttack(self.config_parser, self.attack_config)
        logging.info(f"Checking the config of {self.attack_config['attack_type']}.")
        self.__config_check()
        if self.attack_type == "GIAforNLP":
            self.attack_mode = MyGIAforNLP(
                attack_config=self.attack_config,
                model=kwargs['model'],
                tokenizer=kwargs['tokenizer'],
                train_iter=kwargs['train_iter'],
                distance_func=self.attack_config['distance_func'],
                gradients=kwargs['gradients'],
                config_parser=self.config_parser,
                data_loader=kwargs['data_loader'],
            )
        elif self.attack_type == "AdversarialAttack":
            self.attack_mode = MyTextAttack(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                use_local_model=self.attack_config['use_local_model'],
                use_local_tokenizer=self.attack_config['use_local_tokenizer'],
                use_local_dataset=self.attack_config['use_local_dataset'],
                model_name_or_path=self.attack_config['model_name_or_path'],
                tokenizer_name_or_path=self.attack_config['tokenizer_name_or_path'],
                dataset_name_or_path=self.attack_config['dataset_name_or_path'],
                display_full_info=self.attack_config['display_full_info'],
								defender=self.attack_config['defender'],
            )
        elif self.attack_type == "FET":
            self.attack_mode = FET(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                use_local_model=self.attack_config['use_local_model'],
                use_local_tokenizer=self.attack_config['use_local_tokenizer'],
                model_name_or_path=self.attack_config['model_name_or_path'],
                tokenizer_name_or_path=self.attack_config['tokenizer_name_or_path'],
                dataset_name_or_path=self.attack_config['dataset_name_or_path'],
                display_full_info=self.attack_config['display_full_info'],
            )
        elif self.attack_type == "BackdoorAttack":
            self.attack_mode = MyBackDoorAttack(
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
                sample_metrics=self.attack_config['sample_metrics'],
            )
        elif self.attack_type == "PoisoningAttack":
            self.attack_mode = PoisoningAttack(
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
        elif self.attack_type == "RLMI":
            ppo_config = {}
            for key, value in self.attack_config['ppo_config'].items():
                ppo_config[key] = value
            self.attack_mode = RLMI(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                dataset_name=self.attack_config['dataset_name'],
                model_name=self.attack_config['model_name'],
                seed=(self.attack_config['seed'] if 'seed' in self.attack_config
                      else self.config_parser['general']['random_seed']),
                device=self.device,
                ppo_config=ppo_config,
                seq_length=self.attack_config['seq_length'],
                target_label=self.attack_config['target_label'],
                max_iterations=self.attack_config['max_iterations'],
                min_input_length=self.attack_config['min_input_length'],
                max_input_length=self.attack_config['max_input_length'],
            )
        elif self.attack_type == "ModelStealingAttack":
            self.attack_mode = MyMeaeQ(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                # attack_nums=self.attack_config['attack_nums'],
                # display_full_info=self.attack_config['display_full_info'],
                # use_local_model=self.attack_config['use_local_model'],
                # use_local_tokenizer=self.attack_config['use_local_tokenizer'],
                # use_local_dataset=self.attack_config['use_local_dataset'],
                # model_name_or_path=self.attack_config['model_name_or_path'],
                # tokenizer_name_or_path=self.attack_config['tokenizer_name_or_path'],
                # dataset_name_or_path=self.attack_config['dataset_name_or_path'],
                # query_num=self.attack_config['query_num'],
                # pool_data_type=self.attack_config['pool_data_type'],
                # prompt=self.attack_config['prompt'],
                # pool_subsize=self.attack_config['pool_subsize'],
                # pool_data_source=self.attack_config['pool_data_source'],
                # method=self.attack_config['method'],
                # epsilon=self.attack_config['epsilon'],
                # initial_sample_method=self.attack_config['initial_sample_method'],
                # initial_drk_model=self.attack_config['initial_drk_model'],
                # al_sample_batch_num=self.attack_config['al_sample_batch_num'],
                # al_sample_method=self.attack_config['al_sample_method'],
                # weighted_cross_entropy=self.attack_config['weighted_cross_entropy'],
                # tokenize_max_length=self.attack_config['tokenize_max_length'],
                # batch_size=self.attack_config['batch_size'],
                # optimizer=self.attack_config['optimizer'],
                # learning_rate=self.attack_config['learning_rate'],
                # weight_decay=self.attack_config['weight_decay'],
                # num_epochs=self.attack_config['num_epochs'],
                method=self.attack_config['method'],
                query_num=self.attack_config['query_num'],
                run_seed_arr=self.attack_config['run_seed_arr'],
                pool_data_type=self.attack_config['pool_data_type'],
                pool_data_source=self.attack_config['pool_data_source'],
                pool_subsize=self.attack_config['pool_subsize'],
                prompt=self.attack_config['prompt'],
                epsilon=self.attack_config['epsilon'],
                initial_sample_method=self.attack_config['initial_sample_method'],
                initial_drk_model=self.attack_config['initial_drk_model'],
                al_sample_batch_num=self.attack_config['al_sample_batch_num'],
                al_sample_method=self.attack_config['al_sample_method'],
            )
        elif self.attack_type == "NOP":
            self.attack_mode = NOP(
                config_parser=self.config_parser,
                attack_config=self.attack_config,
                nop_config0=self.attack_config['nop_config0'],
                nop_config1=self.attack_config['nop_config1'],
            )

    def attack(self):
        self.attack_mode.attack()

    def __config_check(self):
        GIAforNLP_config = [
            'optimizer',
            'attack_batch',
            'attack_nums',
            'distance_func',
            'attack_lr',
            'attack_iters',
        ]
        AdversarialAttack_config = [
            'attack_recipe',
            'use_local_model',
            'use_local_tokenizer',
            'use_local_dataset',
            'model_name_or_path',
            'tokenizer_name_or_path',
            'dataset_name_or_path',
            'attack_nums',
            'display_full_info',
            'defender',
        ]
        FET_config = [
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
            'sample_metrics',
        ]
        PoisoningAttack_config = [
            'poisoning_rate',
            'epochs',
            'display_full_info',
        ]
        RLMI_config = [
            'dataset_name',
            'model_name',
            'ppo_config',
            'seq_length',
            'target_label',
            'max_iterations',
            'min_input_length',
            'max_input_length',
            'num_generation',
        ]
        ModelStealingAttack_config = [
            # 'attack_nums',
            # 'display_full_info',
            # 'use_local_model',
            # 'use_local_tokenizer',
            # 'use_local_dataset',
            # 'model_name_or_path',
            # 'tokenizer_name_or_path',
            # 'dataset_name_or_path',
            # 'query_num',
            # 'pool_data_type',
            # 'prompt',
            # 'pool_subsize',
            # 'pool_data_source',
            # 'method',
            # 'epsilon',
            # 'initial_sample_method',
            # 'initial_drk_model',
            # 'al_sample_batch_num',
            # 'al_sample_method',
            # 'weighted_cross_entropy',
            # 'tokenize_max_length',
            # 'batch_size',
            # 'optimizer',
            # 'learning_rate',
            # 'weight_decay',
            # 'num_epochs',
            'method',
            'query_num',
            'run_seed_arr',
            'pool_data_type',
            'pool_data_source',
            'pool_subsize',
            'prompt',
            'epsilon',
            'initial_sample_method',
            'initial_drk_model',
            'al_sample_batch_num',
            'al_sample_method',
        ]
        NOP_config = [
            'nop_config0',
            'nop_config1',
        ]
        if self.attack_type == "GIAforNLP":
            temp_config = GIAforNLP_config
        elif self.attack_type == "AdversarialAttack":
            temp_config = AdversarialAttack_config
        elif self.attack_type == "FET":
            temp_config = FET_config
        elif self.attack_type == "BackdoorAttack":
            temp_config = BackDoorAttack_config
        elif self.attack_type == "PoisoningAttack":
            temp_config = PoisoningAttack_config
        elif self.attack_type == "RLMI":
            temp_config = RLMI_config
        elif self.attack_type == "ModelStealingAttack":
            temp_config = ModelStealingAttack_config
        elif self.attack_type == "NOP":
            temp_config = NOP_config
        else:
            print_red("AttackTypeError: The " + self.attack_type + " is not implemented yet.")
            raise SystemError

        for config in temp_config:
            if config not in self.attack_config:
                print_red("AttackConfigNotFound: Not Found attack_args." + config + " in the config.yaml.")
                raise SystemError
