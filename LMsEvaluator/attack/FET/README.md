# FET: Text reconstruction attack in federated learning by optimizing word permutation and combination

The codes support our paper "FET: Text reconstruction attack in federated learning by optimizing word permutation and combination".

## Experimental Environment

Ubuntu 20.04    Python 3.8   PyTorch 2.0.0  Cuda 11.8

Python packages include:

transformers 4.34.0

datasets 2.14.5

evaluate 0.4.1

rouge_score

deap 1.4.1

## Datasets and models

Datasets and models are loaded from http://huggingface.co.

Datasets - cola, sst2, rotten_tomatoes

Models - huawei-noah/TinyBERT_General_6L_768D, bert-base-uncased

## Parameters

distance_function: l2

population_size: 300 or 500 depending on sentence length

tournsize: 2

crossover_rate: 0.9

Mutation_rate: 0.1

Max_generations: 100 or larger depending on computational resources

halloffame size: 0.1 * population_size











