from attack.base_attack import BaseAttack
from attack.MeaeQ.steal.model_steal.al_steal import my_al_steal
from attack.MeaeQ.steal.model_steal.gen_query import my_gen_query
from utils.defense_utils import output_perturb_defense
import logging


class MyMeaeQ(BaseAttack):
    def __init__(self, config_parser, attack_config, **kwargs):
        super().__init__(config_parser, attack_config)
        self.my_gen_query = my_gen_query(my_args=None, **kwargs)
        self.my_al_steal = my_al_steal(my_args=None, **kwargs)
        self.defender = attack_config.get('defender', None)

    def attack(self):
        logging.info(f"[MeaeQ] 当前attack_config: {self.attack_config}")
        logging.info("[MeaeQ] 开始生成窃取查询...")
        # self.my_gen_query.generate_query()
        logging.info("[MeaeQ] 查询生成完成。")
        if self.defender:
            logging.info(f"[MeaeQ] 检测到defender配置: {self.defender}")
            logging.info("[MeaeQ] [防御] 开始执行主窃取流程...")
            self.my_al_steal.main(defender=self.defender)
            #logging.info(f"[MeaeQ] [防御] 主窃取流程执行完毕。准确率: {acc}, 一致性: {agreement}")
        else:
            logging.info("[MeaeQ] 未检测到defender配置，直接执行主窃取流程...")
            self.my_al_steal.main(defender=None)
            #logging.info(f"[MeaeQ] 主窃取流程执行完毕。准确率: {acc}, 一致性: {agreement}")


if __name__ == '__main__':
    # my_gen_query = my_gen_query()
    # my_gen_query.generate_query()

    # my_al_steal = my_al_steal()
    # my_al_steal.main()
    pass
