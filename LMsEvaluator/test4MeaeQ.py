from attack.MeaeQ.steal.model_steal.gen_query import my_gen_query
from attack.MeaeQ.steal.model_steal.al_steal import my_al_steal
from transformers import AdamW

if __name__ == '__main__':
    my_gen_query = my_gen_query()
    my_gen_query.generate_query()

    my_al_steal = my_al_steal()
    my_al_steal.main()
