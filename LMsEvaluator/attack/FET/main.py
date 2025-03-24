import os
import array
import logging

import torch
import random
import evaluate
import numpy as np
from deap import base, creator, tools
from utils.my_prettytable import MyPrettyTable
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 根据项目调整了引用，其余部分没有改变
import os
import transformers
from attack.FET.fet_utils import *
import attack.FET.elitism as elitism
from attack.base_attack import BaseAttack
from attack.FET.args_factory import get_arguments

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)
toolbox = base.Toolbox()


class FET(BaseAttack):
    def __init__(self, config_parser, attack_config, use_local_model=False, use_local_tokenizer=False,
                 model_name_or_path=None, tokenizer_name_or_path=None, dataset_name_or_path=None,
                 display_full_info=False):
        super().__init__(config_parser, attack_config)

        self.use_local_model = use_local_model
        self.use_local_tokenizer = use_local_tokenizer

        self.project_path = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2])
        self.model_name_or_path = self.__get_name_or_path(self.use_local_model,
                                                          model_name_or_path, "model")
        self.tokenizer_name_or_path = self.__get_name_or_path(self.use_local_tokenizer,
                                                              tokenizer_name_or_path, "tokenizer")
        self.dataset_name_or_path = dataset_name_or_path
        self.display_full_info = display_full_info
        self.my_handlers = logging.getLogger().handlers

    def __get_name_or_path(self, boolean, name_or_path, info):
        if boolean:
            result = os.path.join(self.project_path, name_or_path)
            try:
                if not os.path.exists(result):
                    raise FileNotFoundError(
                        logging.error("FileNotFoundError: No such file or directory: " + result +
                                      ". Please check the path to local " + info + " in my_textattack.py."))
            except Exception as e:
                print(e)
                raise FileNotFoundError
        else:
            result = os.path.join("textattack", name_or_path)
        return result

    def __update_config(self):
        # seed = self.attack_config['seed']
        self.args.dataset = self.dataset_name_or_path
        self.args.n_attacks = self.attack_config['attack_nums']
        self.args.batch_size = self.attack_config['attack_batch']
        self.args.distance_function = self.attack_config['distance_func']
        self.args.population_size = self.attack_config['population_size']
        self.args.tournsize = self.attack_config['tournsize']
        self.args.crossover_rate = self.attack_config['crossover_rate']
        self.args.mutation_rate = self.attack_config['mutation_rate']
        self.args.max_generations = self.attack_config['max_generations']
        self.args.halloffame_size = self.attack_config['halloffame_size']

    def attack(self):
        self.args = get_arguments()
        self.__update_config()
        args = self.args
        logging.info(args)
        model, tokenizer, dataset = self.__get_modelDataTokenizer(args.dataset, args.model)
        # sample data ids
        ids = random.sample(range(len(dataset)), args.batch_size * args.n_attacks)
        # Start the reconstruction attacks
        References, Predictions, word_recovery_rates, edit_distances = [], [], [], []
        count_perfect = 0
        for n in range(args.n_attacks):
            logging.info('----------------------------------------------------------------------')
            logging.info(f'Attack No.{n + 1}:')
            # get original gradients
            ids_batch = ids[n * args.batch_size: (n + 1) * args.batch_size]
            if args.dataset in ['cola', 'sst2']:
                sentences = [dataset[i]['sentence'] for i in ids_batch]
            if args.dataset == 'rotten_tomatoes':
                sentences = [dataset[i]['text'] for i in ids_batch]
            labels = torch.tensor([dataset[i]['label'] for i in ids_batch]).to(device)
            original_batch = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
            original_outputs = model(**original_batch, labels=labels)
            original_gradients = torch.autograd.grad(original_outputs.loss, model.parameters(), create_graph=True,
                                                     allow_unused=True)
            Reference = tokenizer.batch_decode(original_batch['input_ids'])
            logging.info('Reference = ' + str(Reference))
            # print('Reference = ', Reference, flush=True)

            # reconstruct original sentences
            Prediction, Gen = reconstruct(args, model, tokenizer, labels, original_gradients)
            logging.info('Prediction = ' + str(Prediction))
            logging.info('Generations = ' + str(Gen))
            # print('Prediction = ', Prediction, flush=True)
            # print('Generations = ', Gen, flush=True)

            metric = evaluate.load("evaluate/metrics/rouge")
            # metric = evaluate.load("rouge")
            rouge_scores = metric.compute(predictions=Prediction, references=Reference)
            logging.info('Rouge scores: ' + str(rouge_scores))
            # print('Rouge scores: ', rouge_scores)

            original_words, predict_words = set(), set()
            for ref, pre in zip(Reference, Prediction):
                for i in tokenizer.tokenize(ref):
                    original_words.add(i)
                for i in tokenizer.tokenize(pre):
                    predict_words.add(i)
            revealed_word = original_words.intersection(predict_words)
            word_recovery_rate = len(revealed_word) / len(original_words)
            logging.info('Word recovery rate: ' + str(word_recovery_rate))
            # print('Word recovery rate: ', word_recovery_rate)

            edit_distance = 0
            for ref, pre in zip(Reference, Prediction):
                edit_distance += get_edit_distance(ref, pre)
            logging.info('Edit distance: ' + str(edit_distance))
            # print('Edit distance: ', edit_distance)

            if Prediction == Reference:
                count_perfect += 1
                logging.info('If full recovery: True')
            else:
                logging.info('If full recovery: False')

            References += Reference
            Predictions += Prediction
            word_recovery_rates.append(word_recovery_rate)
            edit_distances.append(edit_distance)

        rouge_total = metric.compute(predictions=Predictions, references=References)

        logging.info('Aggregate rouge scores: ' + str(rouge_total))
        logging.info('Full recovery rate: ' + str(count_perfect / args.n_attacks))
        logging.info('Average word recovery rate: ' + str(sum(word_recovery_rates) / args.n_attacks))
        logging.info('Average edit distance: ' + str(sum(edit_distances) / args.n_attacks))

        table = MyPrettyTable()
        table.add_field_names(['FET Attack Results', ''])
        table.add_row(['Aggregate rouge1 scores:', f"{rouge_total['rouge1']:.4f}"])
        table.add_row(['Aggregate rouge2 scores:', f"{rouge_total['rouge2']:.4f}"])
        table.add_row(['Aggregate rougeL scores:', f"{rouge_total['rougeL']:.4f}"])
        table.add_row(['Full recovery rate:', f"{(count_perfect / args.n_attacks):.2%}"])
        table.add_row(['Average word recovery rate:', f"{(sum(word_recovery_rates) / args.n_attacks):.2%}"])
        table.add_row(['Average edit distance:', str(sum(edit_distances) / args.n_attacks)])
        table.set_align('FET Attack Results', 'l')
        table.set_align('', 'l')
        table.print_table()
        table.logging_table()

    def __get_modelDataTokenizer(self, datasetname, modelname):
        # 数据集load需要梯子, 所以先保存到本地再读取
        if datasetname == 'cola':
            # dataset = load_dataset('glue', 'cola')['train']
            # dataset.save_to_disk('datasets/GLUE/cola')
            dataset = load_from_disk("datasets/GLUE/cola")
        elif datasetname == 'sst2':
            # dataset = load_dataset('glue', 'sst2')['train']
            # dataset.save_to_disk('datasets/GLUE/sst2')
            dataset = load_from_disk("datasets/GLUE/sst2")
        elif datasetname == 'rotten_tomatoes':
            # dataset = load_dataset('rotten_tomatoes')['train']
            # dataset.save_to_disk('datasets/Tomatoes')
            dataset = load_from_disk("datasets/Tomatoes")
        # print(dataset)
        # raise SystemError

        if self.attack_config['use_local_model']:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path).to(device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path).to(device)
        model.eval()

        if self.attack_config['use_local_tokenizer']:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        tokenizer.model_max_length = 512

        return model.to(device), tokenizer, dataset


def get_modelDataTokenizer(datasetname, modelname):
    if datasetname == 'cola':
        dataset = load_dataset('glue', 'cola')['train']
    elif datasetname == 'sst2':
        dataset = load_dataset('glue', 'sst2')['train']
    elif datasetname == 'rotten_tomatoes':
        dataset = load_dataset('rotten_tomatoes')['train']
    # print(dataset)
    # raise SystemError

    # dataset = load_from_disk(dataset_path) # load local data

    if modelname == 'tinybert':
        model_path = 'huawei-noah/TinyBERT_General_6L_768D'
    elif modelname == 'bert-base':
        model_path = 'bert-base-uncased'
    elif modelname == 'bert-large':
        model_path = 'bert-large-uncased'
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer.model_max_length = 512

    return model, tokenizer, dataset


def decode_chromosome(mode, individual, idcodes, sentence_length, batch_size):
    if mode == 1:
        ids = torch.tensor([101] + [idcodes[i] for i in individual] + [102]).unsqueeze(0).to(device)
        return ids

    elif mode == 2:
        valid = True
        # check the begin and the end
        if individual[0] >= sentence_length or (individual[-1] >= sentence_length):
            valid = False
        # check the adjacent
        positions = [i for i in range(len(individual)) if individual[i] >= sentence_length]
        for i in range(len(positions) - 1):
            if positions[i] + 1 == positions[i + 1]:
                valid = False

        if valid == False:
            return torch.tensor([101] + [0 for _ in range(sentence_length)] + [102]).unsqueeze(0).to(device)
        else:
            route, routes = [], []
            flag = True
            for i in range(len(individual)):
                if individual[i] < sentence_length:
                    if flag == False:
                        routes.append(route)
                        route = [individual[i]]
                    else:
                        route.append(individual[i])
                    flag = False
                else:
                    flag = True
            if route:
                routes.append(route)

            ids0 = [None for _ in range(sentence_length)]
            for i in range(len(routes)):
                id = idcodes[i]
                for index in routes[i]:
                    ids0[index] = id
            ids = torch.tensor([101] + ids0 + [102]).unsqueeze(0).to(device)
            return ids

    else:
        # check whether valid
        valid = True
        # check the begin and the end
        if individual[0] >= batch_size * sentence_length or (individual[-1] >= batch_size * sentence_length):
            valid = False
        # check the adjacent
        positions = [i for i in range(len(individual)) if individual[i] >= batch_size * sentence_length]
        for i in range(len(positions) - 1):
            if positions[i] + 1 == positions[i + 1]:
                valid = False

        if valid == False:
            ids = []
            for n in range(batch_size):
                ids.append([101] + [0 for _ in range(sentence_length)] + [102])
            return torch.tensor(ids).to(device)

        else:
            route, routes = [], []
            flag = True
            for i in range(len(individual)):
                if individual[i] < batch_size * sentence_length:
                    if flag == False:
                        routes.append(route)
                        route = [individual[i]]
                    else:
                        route.append(individual[i])
                    flag = False
                else:
                    flag = True
            if route:
                routes.append(route)

            valid = True
            tmp = [[] for _ in range(batch_size)]
            for i in routes[0]:  # pad对应的序号
                tmp[i // sentence_length].append(i % sentence_length)
            for i in tmp:
                if i != []:
                    if max(i) != sentence_length - 1:
                        valid = False
                    i.sort()  # 先对数字列表进行排序
                    for j in range(len(i) - 1):
                        if i[j] + 1 != i[j + 1]:
                            valid = False
            if valid == False:
                ids = []
                for n in range(batch_size):
                    ids.append([101] + [0 for _ in range(sentence_length)] + [102])
                return torch.tensor(ids).to(device)

            # else:
            ids0 = [None for _ in range(batch_size * sentence_length)]
            for i in range(len(routes)):
                id = idcodes[i]
                for index in routes[i]:
                    ids0[index] = id
            ids = [[] for _ in range(batch_size)]
            for n in range(batch_size):
                ids[n] = [101] + ids0[n * sentence_length: (n + 1) * sentence_length] + [102]
            return torch.tensor(ids).to(device)


def reconstruct(args, model, tokenizer, labels, original_gradients):
    # infer possible words and sentence length
    id_set = get_id_set(tokenizer, original_gradients[0])
    idcodes = {i: id_set[i] for i in range(len(id_set))}
    sentence_length = get_length(original_gradients[1]) - 2

    # mode 1, 2 or 3 denotes one sentence without repeated words, one sentence with repeated words, or more than one sentence
    if args.batch_size == 1:
        if sentence_length == len(id_set):  # check if including repeated words
            mode = 1
            chromosome_length = len(id_set)
        else:
            mode = 2
            chromosome_length = 2 * sentence_length - len(id_set)
    else:
        mode = 3
        chromosome_length = 2 * args.batch_size * sentence_length - len(idcodes) - 1
        idcodes.clear()
        idcodes = {0: 0}  # add [PAD]
        for i in range(len(id_set)):
            idcodes[i + 1] = id_set[i]

    # fitness function evaluating gradient distance
    def getFitness(individual):
        ids = decode_chromosome(mode, individual, idcodes, sentence_length, args.batch_size)
        attention_mask = torch.tensor([[int(i > 0) for i in id] for id in ids]).to(device)
        outputs = model(input_ids=ids, attention_mask=attention_mask, labels=labels)
        gradients = torch.autograd.grad(outputs.loss, model.parameters())
        distance = l2(gradients, original_gradients).item()
        return distance,

    toolbox.register("randomOrder", random.sample, range(chromosome_length), chromosome_length)
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    toolbox.register("evaluate", getFitness)
    toolbox.register("select", tools.selTournament, tournsize=args.tournsize)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / chromosome_length)
    toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=2.0 / chromosome_length)
    # toolbox.register("mate", tools.cxOrdered)

    population = toolbox.populationCreator(n=args.population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(args.halloffame_size)
    population, logbook, Gen = elitism.eaSimpleWithElitism(population, toolbox,
                                                           cxpb=args.crossover_rate,
                                                           mutpb=args.mutation_rate,
                                                           ngen=args.max_generations,
                                                           stats=stats, halloffame=hof, verbose=True)
    best = hof.items[0]
    reconstructed_ids = decode_chromosome(mode, best, idcodes, sentence_length, args.batch_size)
    Prediction = tokenizer.batch_decode(reconstructed_ids)
    return Prediction, Gen


# 本地测试
if __name__ == "__main__":
    """
    FET模块功能测试
    """
    args = get_arguments()
    logging.info(args)
    model, tokenizer, dataset = get_modelDataTokenizer(args.dataset, args.model)
    # sample data ids
    ids = random.sample(range(len(dataset)), args.batch_size * args.n_attacks)
    # Start the reconstruction attacks
    References, Predictions, word_recovery_rates, edit_distances = [], [], [], []
    count_perfect = 0
    for n in range(args.n_attacks):
        logging.info('=' * 50)
        logging.info(f'Attack No.{n + 1}:')
        # get original gradients
        ids_batch = ids[n * args.batch_size: (n + 1) * args.batch_size]
        if args.dataset in ['cola', 'sst2']:
            sentences = [dataset[i]['sentence'] for i in ids_batch]
        if args.dataset == 'rotten_tomatoes':
            sentences = [dataset[i]['text'] for i in ids_batch]
        labels = torch.tensor([dataset[i]['label'] for i in ids_batch]).to(device)
        original_batch = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        original_outputs = model(**original_batch, labels=labels)
        original_gradients = torch.autograd.grad(original_outputs.loss, model.parameters(), create_graph=True,
                                                 allow_unused=True)
        Reference = tokenizer.batch_decode(original_batch['input_ids'])
        logging.info('Reference = ' + str(Reference))

        # reconstruct original sentences
        Prediction, Gen = reconstruct(args, model, tokenizer, labels, original_gradients)
        logging.info('Prediction = ' + str(Prediction))
        logging.info('Generations = ' + str(Gen))

        metric = evaluate.load("evaluate/metrics/rouge")
        # metric = evaluate.load("rouge")
        rouge_scores = metric.compute(predictions=Prediction, references=Reference)
        logging.info('Rouge scores: ' + str(rouge_scores))

        original_words, predict_words = set(), set()
        for ref, pre in zip(Reference, Prediction):
            for i in tokenizer.tokenize(ref):
                original_words.add(i)
            for i in tokenizer.tokenize(pre):
                predict_words.add(i)
        revealed_word = original_words.intersection(predict_words)
        word_recovery_rate = len(revealed_word) / len(original_words)
        logging.info('Word recovery rate: ' + str(word_recovery_rate))

        edit_distance = 0
        for ref, pre in zip(Reference, Prediction):
            edit_distance += get_edit_distance(ref, pre)
        logging.info('Edit distance: ' + str(edit_distance))

        if Prediction == Reference:
            count_perfect += 1
            logging.info('If full recovery: True')
        else:
            logging.info('If full recovery: False')

        References += Reference
        Predictions += Prediction
        word_recovery_rates.append(word_recovery_rate)
        edit_distances.append(edit_distance)

    rouge_total = metric.compute(predictions=Predictions, references=References)
    logging.info('Aggregate rouge scores: ' + str(rouge_total))
    logging.info('Full recovery rate: ' + str(count_perfect / args.n_attacks))
    logging.info('Average word recovery rate: ' + str(sum(word_recovery_rates) / args.n_attacks))
    logging.info('Average edit distance: ' + str(sum(edit_distances) / args.n_attacks))

    table = MyPrettyTable()
    table.add_field_names(['FET Attack Results', ''])
    table.add_row(['Aggregate rouge1 scores:', f"{rouge_total['rouge1']:.4f}"])
    table.add_row(['Aggregate rouge2 scores:', f"{rouge_total['rouge2']:.4f}"])
    table.add_row(['Aggregate rougeL scores:', f"{rouge_total['rougeL']:.4f}"])
    table.add_row(['Full recovery rate:', f"{(count_perfect / args.n_attacks):.2%}"])
    table.add_row(['Average word recovery rate:', f"{(sum(word_recovery_rates) / args.n_attacks):.2%}"])
    table.add_row(['Average edit distance:', str(sum(edit_distances) / args.n_attacks)])
    table.set_align('FET Attack Results', 'l')
    table.set_align('', 'l')
    table.print_table()
    table.logging_table()
