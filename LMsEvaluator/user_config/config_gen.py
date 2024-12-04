import yaml,os

# 全局攻击策略
AdvAttack = {
    "attack": True,
    "attack_type": "AdvAttack",
    "attack_recipe": "TextFoolerJin2019",
    "use_local_model": True,
    "use_local_tokenizer": True,
    "use_local_dataset": True,
    "model_name_or_path": "LMs/bert_base_uncased_english",
    "tokenizer_name_or_path": "LMs/bert_base_uncased_english",
    "dataset_name_or_path": "datasets/imdb/train.txt",
    "attack_nums": 2,
    "display_full_info": True
}

SWAT = {
    "attack": False,
    "attack_type": "SWAT",
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
    "model_name_or_path": "LMs/bert_base_uncased_english",
    "tokenizer_name_or_path": "LMs/bert_base_uncased_english",
    "dataset_name_or_path": "cola",
    "display_full_info": True
}

BackDoorAttack = {
    "attack": True,
    "attack_type": "BackDoorAttack",
    "use_local_model": True,
    "model": "bert",
    "model_name_or_path": "LMs/bert_base_uncased_english",
    "poison_dataset": "sst-2",
    "target_dataset": "sst-2",
    "poisoner": {"name": "badnets"},
    "train": {"name": "base", "batch_size": 16, "epochs": 1},
    "defender": "None",
    "sample_metrics": [],
    "display_full_info": True
}

PoisoningAttack = {
    "attack": False,
    "attack_type": "PoisoningAttack",
    "poisoning_rate": 0.1,
    "epochs": 10,
    "display_full_info": True
}

# 攻击策略字典
attack_strategies = {
    "AdvAttack": AdvAttack,
    "SWAT": SWAT,
    "BackDoorAttack": BackDoorAttack,
    "PoisoningAttack": PoisoningAttack
}

def generate_config(username, attack_types):
    config = {
        "general": {
            "random_seed": 0,
            "use_gpu": True
        },
        "LM_config": {
            "model": "bert_base_uncased_english"
        },
        "task_config": {
            "task": "TaskForSingleSentenceClassification",
            "dataset": "imdb",
            "dataset_type": ".txt",
            "split_sep": "_!_",
            "epochs": 1
        },
        "attack_list": [],
        "output": {
            "base_path": "output",
            "model_output": "modelOutput",
            "evaluation_result": "evaluationResult"
        }
    }

    # 遍历输入的 attack_types，修改配置
    for attack_info in attack_types:
        attack_type = attack_info['attack_type']
        index = attack_info['index']
        attack = attack_info['attack']

        # 检查攻击类型是否在已定义的策略中
        if attack_type in attack_strategies:
            attack_config = attack_strategies[attack_type].copy()  # 复制攻击策略

            # 根据输入修改攻击配置
            attack_config["attack"] = attack

            # 根据攻击类型的不同修改攻击配置
            if attack_type == "AdvAttack" and attack:
                attack_config["attack_recipe"] = attack_info.get('attack_recipe', "TextFoolerJin2019")

            # 将修改后的配置添加到attack_list
            if len(config["attack_list"]) <= index:
                config["attack_list"].append({"attack_args": attack_config})
            else:
                config["attack_list"][index]["attack_args"] = attack_config

    # 将配置保存到文件
    output_dir = "user_config"
    file_name = os.path.join(output_dir, f"{username}_config.yaml")
    with open(file_name, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

    print(f"Config file for {username} generated: {file_name}")

# 示例输入
'''
attack_types = [
    {'index': 0, 'attack': True, 'attack_type': 'AdvAttack', 'attack_recipe': 'TextFoolerJin2019'},
    {'index': 1, 'attack': True, 'attack_type': 'SWAT', 'attack_recipe': 'default'}
]
generate_config("user123", attack_types)
'''



