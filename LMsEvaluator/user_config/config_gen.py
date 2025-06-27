import yaml, os
import copy  # 引入 copy 模块
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
    "sample_metrics": [],# ['ppl', 'use', 'grammar']
    "display_full_info": True,
    "defender": "None" # str: 所选择的防御方法[BKI,ONION,STRIP,RAP,CUBE]
}

PoisoningAttack = {
    "attack": True,
    "attack_type": "PoisoningAttack",
    "poisoning_rate": 0.1,
    "epochs": 10,
    "display_full_info": True
}
RLMI={
    "attack": False,
    "attack_type": "RLMI",
    "dataset_name": "emotion",
    "model_name": "tinybert4",
    "seed": 42,
    "ppo_config":{
        "mini_batch_size": 16,
        "batch_size": 16,
        "log_with": "None", #log_with: "wandb" # tensorboard, wandb
        "learning_rate": 1e-5
    },
    "seq_length": 20,
    "target_label": 0,
    "max_iterations": 2000,
    "min_input_length": 2,
    "max_input_length": 5,
    "num_generation": 1000
}
GIAforNLP={
    "attack": True,
    "attack_type": "GIAforNLP",
    "attack_data": "None",
    "optimizer": "Adam",
    "attack_batch": 2,
    "attack_nums": 1,
    "distance_func": "l2",             # str: 攻击方法中的距离函数，'l2' or 'cos'
    "attack_lr": 0.01,                 # float: 攻击方法中的学习率
    "attack_iters": 10,                # int: 一次攻击中的迭代轮次
    "display_full_info": True         # boolean: 是否显示全部过程信息
}

# 攻击策略字典
attack_strategies = {
    "AdvAttack": AdvAttack,
    "FET": FET,
    "BackDoorAttack": BackDoorAttack,
    "PoisoningAttack": PoisoningAttack,
    "RLMI": RLMI,
    "GIAforNLP": GIAforNLP
}


def generate_config(username, attack_types):
    # 初始化配置字典
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
            "dataset_type": '.txt',
            "split_sep": '_!_',
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
    for index, attack_info in enumerate(attack_types):
        attack_type = attack_info.get('type')

        # 检查攻击类型是否在已定义的策略中
        if attack_type in attack_strategies:
            attack_config = copy.deepcopy(attack_strategies[attack_type])  # 深度复制攻击策略
            if "attack_recipe" in attack_config:
                attack_config['attack_recipe']=attack_info['strategy']
            for key in attack_config:
                if key in attack_info['params']:
                    attack_config[key] = attack_info['params'][key]

            # 将修改后的配置添加到 attack_list
            if len(config["attack_list"]) <= index:
                config["attack_list"].append({"attack_args": attack_config})
            else:
                config["attack_list"][index]["attack_args"] = attack_config

    # 获取当前脚本的绝对路径，并保存到当前目录
    output_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 确保输出目录存在

    # 生成配置文件的路径
    file_name = os.path.join(output_dir, f"{username}_config.yaml")

    # 将配置保存到 YAML 文件

    with open(file_name, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

    print(f"Config file for {username} generated: {file_name}")


# 示例输入
if __name__ == '__main__':

    username = "test_user"
    attack_types = [
        {
            "type": "AdvAttack",
            "strategy": "TextFoolerJin2019",
            "params": {
                "attack_nums": 2,
                "display_full_info": True
            }
        },
        {
            "type": "FET",
            "strategy": "FETStrategy",
            "params": {
                "population_size": 300,
                "max_generations": 2
            }
        },
        {
            "type": "BackDoorAttack",
            "strategy": "",
            "params": {
                "poisoning_rate": 0.1,
                "epochs": 10
            }
        }
    ]

    generate_config(username, attack_types)




