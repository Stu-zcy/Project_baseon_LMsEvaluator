import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='SWAT attack')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rotten_tomatoes'], default='cola')
    parser.add_argument('--model', choices=['tinybert', 'bert-base', 'bert-large'], default='tinybert')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_attacks', type=int, default=100)

    # parameters of the genetic algorithm
    parser.add_argument('--distance_function', choices=['cos', 'l2'], default='l2')
    parser.add_argument('--population_size', type=int, default=300)
    parser.add_argument('--tournsize', type=int, default=2)
    parser.add_argument('--crossover_rate', type=float, default=0.9)
    parser.add_argument('--mutation_rate', type=float, default=0.1)
    parser.add_argument('--max_generations', type=int, default=100)
    parser.add_argument('--halloffame_size', type=int, default=30)
    args = parser.parse_args()

    return args
