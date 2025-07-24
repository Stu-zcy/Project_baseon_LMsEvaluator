import yaml
import os
import copy

# 全局攻击策略
AdvAttack = {
    "attack": True,
    "attack_type": "AdversarialAttack",
    "attack_recipe": "TextFoolerJin2019",
    "use_local_model": True,
    "use_local_tokenizer": True,
    "use_local_dataset": True,
    "model_name_or_path": "LMs/bert_base_uncased",
    "tokenizer_name_or_path": "LMs/bert_base_uncased",
    "dataset_name_or_path": "datasets/imdb/train.txt",
    "attack_nums": 2,
    "display_full_info": True,
    "defender": {
        "num_epochs": 1,
        "num_clean_epochs": 1,
        "num_train_adv_examples": 1000,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "log_to_tb": False
    }
}

FET = {
    "attack": True,
    "attack_type": "FET",
    "seed": 42,
    "attack_batch": 2,
    "attack_nums": 1,
    "distance_func": "l2",
    "population_size": 300,
    "tournsize": 5,
    "crossover_rate": 0.9,
    "mutation_rate": 0.1,
    "max_generations": 2,
    "halloffame_size": 30,
    "use_local_model": True,
    "use_local_tokenizer": True,
    "model_name_or_path": "LMs/bert_base_uncased",
    "tokenizer_name_or_path": "LMs/bert_base_uncased",
    "dataset_name_or_path": "cola",
    "display_full_info": True
}

BackDoorAttack = {
    "attack": True,
    "attack_type": "BackDoorAttack",
    "use_local_model": True,
    "model": "bert",
    "model_name_or_path": "LMs/bert_base_uncased",
    "poison_dataset": "sst-2",
    "target_dataset": "sst-2",
    "poisoner": {"name": "badnets"},
    "train": {"name": "base", "batch_size": 16, "epochs": 1},
    "sample_metrics": [],  # ['ppl', 'use', 'grammar']
    "display_full_info": True,
    "defender": "None"  # str: 所选择的防御方法[BKI,ONION,STRIP,RAP,CUBE]
}

PoisoningAttack = {
    "attack": True,
    "attack_type": "PoisoningAttack",
    "poisoning_rate": 0.1,
    "save_path": './attack/PoisoningAttack/model_output',
    "train_config": {
        "output_dir": './attack/PoisoningAttack/cache',
        "num_train_epochs": 1,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 64,
        "warmup_steps": 1000,
        "weight_decay": 0.01,
        "logging_dir": './logs',
        "logging_steps": 1000,
        "run_name": 'my_experiment',
        "report_to": "none"
    }
}

RLMI = {
    "attack": False,
    "attack_type": "RLMI",
    "dataset_name": "emotion",
    "model_name": "tinybert4",
    "seed": 42,
    "ppo_config": {
        "mini_batch_size": 16,
        "batch_size": 16,
        "log_with": "None",  # log_with: "wandb" # tensorboard, wandb
        "learning_rate": 1e-5
    },
    "seq_length": 20,
    "target_label": 0,
    "max_iterations": 2000,
    "min_input_length": 2,
    "max_input_length": 5,
    "num_generation": 1000
}



ModelStealingAttack = {
    "attack": True,
    "attack_type": "ModelStealingAttack",
    "method": "RS",
    "query_num": 320,
    "run_seed_arr": [56],
    "pool_data_type": "whole",
    "pool_data_source": "wiki",
    "pool_subsize": -1,
    "prompt": None,
    "epsilon": -1,
    "initial_sample_method": "random_sentence",
    "initial_drk_model": None,
    "al_sample_batch_num": -1,
    "al_sample_method": "random"
}

# 攻击策略字典
attack_strategies = {
    "AdversarialAttack": AdvAttack,
    "FET": FET,
    "BackDoorAttack": BackDoorAttack,
    "PoisoningAttack": PoisoningAttack,
    "RLMI": RLMI,
    "ModelStealingAttack": ModelStealingAttack
}


def generate_config(username, attack_list, globalConfig=None):
    """
    生成配置文件
    
    参数:
    username (str): 用户名
    attack_list (list): 攻击配置列表，每个元素是一个字典
    """
    # 初始化配置字典
    config = {
        "general": {
            "random_seed": 0,
            "use_gpu": True,
            "log_file_name": 'single',
            "logs_save_dir": './logs'
        },
        "LM_config": {
            "model": "bert_base_uncased",
            "local_model": True
        },
        "task_config": {
            "task": "TaskForSingleSentenceClassification",
            "dataset": "imdb",
            "local_dataset": True,
            "normal_training": False,
            "save_path": './cache/model_output',
            "train_config": {
                "output_dir": './cache',
                "num_train_epochs": 1,
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 64,
                "warmup_steps": 1000,
                "weight_decay": 0.01,
                "logging_dir": './logs',
                "logging_steps": 1000,
                "run_name": 'my_experiment',
                "report_to": "none"
            },
        },
        "attack_list": [],
        "output": {
            "base_path": "output",
            "model_output": "modelOutput",
            "evaluation_result": "evaluationResult"
        }
    }
    # 如果提供了全局配置，则合并到配置中
    if globalConfig:
        # 合并 general
        if 'general' in globalConfig:
            config['general']['random_seed'] = globalConfig['general']['random_seed']
            config['general']['use_gpu'] = globalConfig['general']['use_gpu']
            config['general']['log_file_name'] = globalConfig['general']['log_file_name']
        # 合并模型配置
        if 'model' in globalConfig:
            config['LM_config']['model'] = globalConfig['model']['predefined']
            config['LM_config']['local_model'] = globalConfig['model']['local_model']
        # 合并任务配置
        if 'task' in globalConfig:
            config['task_config']['task'] = globalConfig['task']['task']
            config['task_config']['dataset'] = globalConfig['task']['dataset']
            config['task_config']['local_dataset'] = globalConfig['task']['local_dataset']
            config['task_config']['normal_training'] = globalConfig['task']['normal_training']

    # 遍历输入的攻击列表，修改配置
    for attack in attack_list:
        if attack.get('status') != 'active':
            continue  # 跳过非活跃配置

        attack_type = attack.get('type')
        attack_name = attack.get('name')
        created_at = attack.get('createdAt')
        
       
        # 检查攻击类型是否在已定义的策略中
        if attack_type in attack_strategies:
            # 深度复制攻击策略
            attack_config = copy.deepcopy(attack_strategies[attack_type])
            
            # 更新攻击策略
            if "attack_recipe" in attack_config:
                attack_config['attack_recipe'] = attack.get('strategy', 'default')
            
            # 更新参数
            params = attack.get('params', {})
            for key in params:
                if key in attack_config:
                    attack_config[key] = params[key]
            
            # 将修改后的配置添加到 attack_list
            config["attack_list"].append({
                "attack_args": attack_config
            })

    # 获取当前脚本的绝对路径，并保存到当前目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成配置文件的路径
    config_file = os.path.join(output_dir, f"{username}_config.yaml")
    
    # 将配置保存到 YAML 文件
    with open(config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    attack_info=[attack_list,globalConfig]
    # 生成攻击信息文件
    info_file = os.path.join(output_dir, f"{username}_attack_info.yaml")
    with open(info_file, 'w') as file:
        yaml.dump(attack_info, file, default_flow_style=False, allow_unicode=True)
    print(f"配置文件已生成: {config_file}")
    print(f"攻击信息文件已生成: {info_file}")

def get_attack_info(username):
    """
    根据用户名获取攻击信息
    
    参数:
    username (str): 用户名
    
    返回:
    list: 攻击信息列表，如果文件不存在则返回空列表
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))
    info_file = os.path.join(output_dir, f"{username}_attack_info.yaml")
    
    if not os.path.exists(info_file):
        print(f"攻击信息文件不存在: {info_file}")
        return []
    
    try:
        with open(info_file, 'r') as file:
            attack_info = yaml.safe_load(file)
            return attack_info if attack_info else []
    except Exception as e:
        print(f"读取攻击信息文件失败: {e}")
        return []

    
def update_log_file_name(username, new_log_file_name):
    """
    更新日志文件名
    
    参数:
    username (str): 用户名
    new_log_file_name (str): 新的日志文件名
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(output_dir, f"{username}_config.yaml")
    
    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return
    
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        config['general']['log_file_name'] = new_log_file_name
        
        with open(config_file, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        
        print(f"日志文件名已更新为: {new_log_file_name}")
    except Exception as e:
        print(f"更新日志文件名失败: {e}")

if __name__ == '__main__':
    # 示例输入数据
    attack_list = [
        {
            'id': 1,
            'name': '文本对抗攻击配置',
            'type': 'AdversarialAttack',
            'strategy': 'TextFoolerJin2019',
            'description': '针对NLP模型的文本对抗攻击，通过替换同义词和干扰字符欺骗模型',
            'status': 'active',
            'sent': True,
            'executed': True,
            'createdAt': '2025-05-15 10:30',
            'params': {'use_local_model': True, 'attack_nums': 3}
        },
        {
            'id': 3,
            'name': '数据投毒配置',
            'type': 'PoisoningAttack',
            'strategy': 'default',
            'description': '针对训练数据的投毒攻击，污染训练集以降低模型性能',
            'status': 'active',
            'sent': True,
            'executed': False,
            'createdAt': '2025-05-20 09:45',
            'params': {'poisoning_rate': 0.15, 'epochs': 15}
        },
        {
            'id': 4,
            'name': '模型窃取攻击',
            'type': 'ModelStealingAttack',
            'strategy': 'default',
            'description': '通过查询目标模型来窃取其结构和参数',
            'status': 'active',
            'sent': True,
            'executed': True,
            'createdAt': '2025-06-10 14:30',
            'params': {'attack_nums': 3, 'query_num': 500, 'method': 'MeaeQ'}
        },
        {
            'id': 5,
            'name': '模型反演配置',
            'type': 'RLMI',
            'strategy': 'default',
            'description': '使用强化学习优化攻击策略',
            'status': 'active',
            'sent': True,
            'executed': False,
            'createdAt': '2025-06-12 11:20',
            'params': {'seed': 123, 'max_iterations': 2500}
        }
    ]

    username = "test_user"
    generate_config(username, attack_list)
    
   