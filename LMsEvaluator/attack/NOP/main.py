import os
import logging
from attack.base_attack import BaseAttack


class NOP(BaseAttack):
    def __init__(self, config_parser, attack_config, nop_config0=None, nop_config1=None):
        super().__init__(config_parser, attack_config)

    def attack(self):
        logging.info("NOP Attack执行结束。")
        print("NOP Attack执行结束。")


# 本地测试
if __name__ == "__main__":
    """
    NOP模块功能测试
    """

    # 项目路径获取
    projectPath = os.path.dirname(os.path.abspath(__file__))
    projectPath = "/".join(projectPath.split("/")[:-2])

    attack_mode = NOP(config_parser={}, attack_config={})
    attack_mode.attack()
