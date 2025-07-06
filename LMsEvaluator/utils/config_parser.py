import os
import yaml
from utils.my_exception import print_red
from tasks import TaskForSingleSentenceClassification, TaskForSQuADQuestionAnswering, TaskForChineseNER,TaskForMultipleChoice, TaskForPairSentenceClassification, TaskForPretraining#,TaskForSequenceClassification


def parse_config(projectPath, initTime, username='default'):
    """
    配置文件检查 + 配置解析器生成 + 下游任务初始化 + 攻击模块初始化
    :param projectPath: 项目根目录
    :return: modelClass: 下游任务模块
    """
    # 读取配置文件
    if username =='default':
        with open('config.yaml', 'r', encoding='utf-8') as configFile:
            configParser = yaml.load(configFile, Loader=yaml.FullLoader)
    else:
        file_name = os.path.join("user_config", f"{username}_config.yaml")
        with open(file_name, 'r', encoding='utf-8') as configFile:
            configParser = yaml.load(configFile, Loader=yaml.FullLoader)
    # 配置信息测试 + 项目路径测试
    # print(configParser)
    # modelOutputPath = configParser['output']['basePath'] + "/" + configParser['output']['modelOutput']
    # print(modelOutputPath)

    # 检查配置文件是否完整
    check_base_config(configParser)

    # 读取常规训练配置
    generalConfig = configParser['general']

    # 读取模型配置
    LMConfig = configParser['LM_config']
    model = LMConfig['model']
    modelPath = os.path.join(projectPath, "LMs", model)

    # 读取数据集配置
    taskConfig = configParser['task_config']
    dataset = taskConfig['dataset']
    datasetType = taskConfig['dataset_type']
    dataPath = os.path.join(projectPath, "datasets", dataset)
    if "split_sep" in taskConfig:
        splitSep = taskConfig['split_sep']
    else:
        splitSep = "_!_"

    # 数据集与预训练模型路径检查
    try:
        if not os.path.exists(dataPath):
            raise FileNotFoundError(
                print_red("FileNotFoundError: No such file or directory: " + str(dataPath) +
                          ". Please check the task_config.dataset config in config.yaml."))
        if not os.path.exists(modelPath):
            raise FileNotFoundError(
                print_red("FileNotFoundError: No such file or directory: " + str(modelPath) +
                          ". Please check the task_config.dataset config in config.yaml."))
    except Exception as e:
        print(e)
        raise SystemError

    # 下游任务检查与初始化
    if taskConfig['task'] == "TaskForSingleSentenceClassification":
        modelClass = TaskForSingleSentenceClassification.TaskForSingleSentenceClassification(
            initTime, dataset, model, datasetType, generalConfig['use_gpu'], splitSep, configParser,
						username=username)
    elif taskConfig['task'] == "TaskForSQuADQuestionAnswering":
        modelClass = TaskForSQuADQuestionAnswering.TaskForSQuADQuestionAnswering(
            initTime, dataset, model, datasetType, generalConfig['use_gpu'], configParser,
						username=username)
    elif taskConfig['task'] == "TaskForChineseNER":
        modelClass = TaskForChineseNER.TaskForChineseNER(initTime, dataset, model, datasetType,
                                                         generalConfig['use_gpu'], splitSep, configParser,
                                                         username=username)
    elif taskConfig['task'] == "TaskForMultipleChoice":
        modelClass = TaskForMultipleChoice.TaskForMultipleChoice(initTime, dataset, model,
                                                                 datasetType, generalConfig['use_gpu'], configParser, 
                                                                 username=username)
    elif taskConfig['task'] == "TaskForPairSentenceClassification":
        modelClass = TaskForPairSentenceClassification.TaskForPairSentenceClassification(initTime, dataset,
                                                                                         model, datasetType,
                                                                                         generalConfig['use_gpu'],
                                                                                         splitSep, configParser,
                                                                                         username=username)
    elif taskConfig['task'] == "TaskForPretraining":
        if dataset == "SongCi":
            modelClass = TaskForPretraining.TaskForPretraining(initTime, dataset, model, datasetType, generalConfig['use_gpu'],
                                                               "songci", configParser, username=username)
        elif dataset == "WikiText":
            modelClass = TaskForPretraining.TaskForPretraining(initTime, dataset, model, datasetType, generalConfig['use_gpu'],
                                                               "wiki2", configParser, username=username)
        else:
            print_red("Please check the 'task_config.dataset' config in config.yaml.")
            print_red("P.S. If the task is TaskForPretraining, then the dataset must be songci or wiki.")
            raise SystemError

    else:
        print_red("ERROR: UNKNOWN TASK.")
        print_red("Please check the 'taskConfig.task' config in config.yaml.")
        raise SystemError

    # 攻击模块配置检查与初始化
    if 'attack_list' in configParser and configParser['attack_list'] is not None:
        configParser['attack_list'] = check_attack_config(configParser['attack_list'], projectPath)
    else:
        configParser['attack_list'] = []

    return modelClass


def check_base_config(config_parser):
    """
    检查config.yaml中基础配置是否存在
    :param config_parser: 配置解析器
    :return: None
    """
    base_config_list = [
        'general',
        'LM_config',
        'task_config',
        'output',
    ]
    general_config_list = [
        'use_gpu',
        'random_seed',
    ]
    LMConfig_config_list = [
        'model',
    ]
    taskConfig_config_list = [
        'task',
        'dataset',
        'dataset_type',
        'split_sep',
        'epochs',
    ]
    base_config_item_list = [general_config_list, LMConfig_config_list, taskConfig_config_list, []]

    for item in base_config_list:
        if item not in config_parser:
            print_red(f"Please check the '{item}' config in config.yaml.")
            raise SystemError

    for base_config, item_list in zip(base_config_list, base_config_item_list):
        if not check_item_in_list(config_parser[base_config], item_list):
            print_red(f"Please check the '{base_config}.{item_list}' config in config.yaml.")
            raise SystemError


def check_item_in_list(full_list, item_list):
    """
    检查full_list中是否含有全部item_list元素
    :param full_list: 父list
    :param item_list: 子list
    :return: boolean
    """
    for item in item_list:
        if item not in full_list:
            return False
    return True


def check_attack_config(attack_list, project_path):
    """
    删除攻击模块中路径错误 + 设置错误的配置
    :param attack_list: [attack_args 1, ..., attack_args n]
    :param project_path: 项目根目录
    :return: attack_list: 删除无效信息后的attack_list
    """
    attack_type_list, del_index = ['AdversarialAttack', 'BackdoorAttack', 'PoisoningAttack', 'FET', 'RLMI', 'GIAforNLP',
                                   'ModelStealingAttack', 'NOP'], []
    for index in range(len(attack_list)):
        # 读取攻击模块配置
        attack_config = attack_list[index]['attack_args']
        if 'attack' in attack_config and attack_config['attack']:
            attack_type = attack_config['attack_type']
            if attack_type == 'ModelStealingAttack':
                attack_path = os.path.join(project_path, "attack", "MeaeQ")
            else:
                attack_path = os.path.join(project_path, "attack", attack_type)

            if attack_type not in attack_type_list:
                print_red("UNKNOWN ATTACK TYPE CONFIG: " + str(attack_type) +
                          ". Please check the attack_args.attack_type config in config.yaml.")
                del_index.append(index)
                continue

            # 攻击模块路径检查
            try:
                if not os.path.exists(str(attack_path)):
                    raise FileNotFoundError(
                        print_red("FileNotFoundError: No such file or directory: " + str(attack_path) +
                                  ". Please check the attack_args.attack_type config in config.yaml."))
            except Exception as e:
                print(e)
                del_index.append(index)
        else:
            del_index.append(index)

    # 删除配置错误的攻击模块
    for index in del_index[::-1]:
        del attack_list[index]

    return attack_list
