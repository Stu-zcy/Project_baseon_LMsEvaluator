import os
import torch

from attack.base_attack import BaseAttack

from utils.my_exception import print_red
from attack.RLMI.get_dataset import get_ori_dataset
from attack.RLMI.get_target_model import get_target_model
from attack.RLMI.train_attack_model import train_attack_model
from attack.RLMI.infer_attack_model import infer_attack_model
from attack.RLMI.evaluate_result import evaluate_result
from attack.RLMI.evaluate import evaluate

test_ppo_config = {
    "mini_batch_size": 16,
    "batch_size": 16,
    # "log_with":'wandb', # tensorboard, wandb
    "log_with": None,
    "learning_rate": 1e-5,
}


class RLMI(BaseAttack):
    def __init__(self, config_parser, attack_config, dataset_name="emotion", model_name="tinybert4", seed=42,
                 device=torch.device('cpu'), ppo_config=None, seq_length=20, target_label=0, max_iterations=2000,
                 min_input_length=2, max_input_length=5, num_generation=1000):
        super().__init__(config_parser, attack_config)
        if ppo_config is None:
            self.ppo_config = test_ppo_config
        else:
            self.ppo_config = ppo_config
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.seed = seed
        self.device = device
        self.seq_length = seq_length
        self.target_label = target_label
        self.max_iterations = max_iterations
        self.min_input_length = min_input_length
        self.max_input_length = max_input_length
        self.num_generation = num_generation

        self.__config_check()

    def __config_check(self):
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        rlmi_attack_path = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(project_path, "LMs", self.model_name)
        dataset_path = os.path.join(project_path, "datasets", self.dataset_name)

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    print_red("FileNotFoundError: No such file or directory: " + model_path +
                              ". Please check the model_name in the config.yaml of RLMI."))
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    print_red("FileNotFoundError: No such file or directory: " + dataset_path +
                              ". Please check the dataset_name in the config.yaml of RLMI."))
        except Exception as e:
            print(e)
            raise SystemError

    def attack(self):
        get_ori_dataset(dataset_name=self.dataset_name, seed=self.seed)

        get_target_model(model_name=self.model_name, dataset_name=self.dataset_name, device=self.device)

        train_attack_model(seed=self.seed, model_name=self.model_name, dataset_name=self.dataset_name, ppo_config=None,
                           seq_length=self.seq_length, target_label=self.target_label,
                           max_iterations=self.max_iterations, min_input_length=self.min_input_length,
                           max_input_length=self.max_input_length, device=self.device)

        infer_attack_model(seed=self.seed, model_name=self.model_name, dataset_name=self.dataset_name,
                           seq_length=self.seq_length, target_label=self.target_label,
                           min_input_length=self.min_input_length, max_input_length=self.max_input_length,
                           num_generation=self.num_generation, device=self.device)

        evaluate_result()
        # evaluate()
