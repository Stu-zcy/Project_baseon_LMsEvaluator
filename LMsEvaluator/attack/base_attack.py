class BaseAttack:
    def __init__(self, config_parser, attack_config):
        self.config_parser = config_parser
        self.attack_config = attack_config

    def attack(self):
        print("请先实例化具体的攻击模块")
