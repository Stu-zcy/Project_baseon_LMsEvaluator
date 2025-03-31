import user_config.config_gen
from utils.config_parser import parse_config

attack_types = [{"attack": True,
    "attack_type": "AdversarialAttack",
    "attack_recipe": "TextFoolerJin2019",
    "use_local_model": True,
    "use_local_tokenizer": True,
    "use_local_dataset": True,
    "model_name_or_path": "LMs/bert_base_uncased_english",
    "tokenizer_name_or_path": "LMs/bert_base_uncased_english",
    "dataset_name_or_path": "datasets/imdb/train.txt",
    "attack_nums": 2,
    "display_full_info": True}, {"attack": True,
    "attack_type": "AdversarialAttack",
    "attack_recipe": "TextFoolerJin2019",
    "use_local_model": True,
    "use_local_tokenizer": True,
    "use_local_dataset": True,
    "model_name_or_path": "LMs/bert_base_uncased_english",
    "tokenizer_name_or_path": "LMs/bert_base_uncased_english",
    "dataset_name_or_path": "datasets/imdb/train.txt",
    "attack_nums": 2,
    "display_full_info": True}]
#user_config.config_gen.generate_config("user123", attack_types)
model_class = parse_config("E:/Desktop/Project/LMsEvaluator/user_config/admin_config.yaml")
