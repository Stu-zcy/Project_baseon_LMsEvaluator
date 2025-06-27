from attack.base_attack import BaseAttack
from attack.MeaeQ.steal.model_steal.gen_query import my_gen_query
from attack.MeaeQ.steal.model_steal.al_steal import my_al_steal


class MyMeaeQ(BaseAttack):
    def __init__(self, config_parser, attack_config, **kwargs):
        super().__init__(config_parser, attack_config)
        self.my_gen_query = my_gen_query(my_args=None, **kwargs)
        self.my_al_steal = my_al_steal(my_args=None, **kwargs)

    def attack(self):
        self.my_gen_query.generate_query()
        self.my_al_steal.main()


if __name__ == '__main__':
    pass
    # my_gen_query = my_gen_query()
    # my_gen_query.generate_query()

    # my_al_steal = my_al_steal()
    # my_al_steal.main()
