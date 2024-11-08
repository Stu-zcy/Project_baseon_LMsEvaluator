import random

from datasets import load_dataset


def get_imdb_dataset():
    dataset = load_dataset("imdb")
    # print(dataset)

    base_path = "../datasets/imdb/"
    # write_one_sentence_to_file(dataset, "train", base_path, "poisoning_train.txt", 'text', "label", "_!_", True, 0.1)
    write_one_sentence_to_file(dataset, "train", base_path, "train.txt", 'text', "label", "_!_")
    write_one_sentence_to_file(dataset, "test", base_path, "test.txt", 'text', "label", "_!_")
    write_one_sentence_to_file(dataset, "test", base_path, "val.txt", 'text', "label", "_!_")


def get_glue_dataset():
    dataset_name = "mrpc"  # cola, mnli, sst2, ax, stsb, wnli, qqp, mrpc...
    dataset = load_dataset("glue", dataset_name)
    # print(dataset)

    base_path = "../datasets/GLUE/" + dataset_name + "/"
    if dataset_name == "cola" or dataset_name == "sst2":
        # write_one_sentence_to_file(dataset, "train", base_path, "poisoning_train.txt", "sentence", "label", "_!_", True,
        #                            0.1)
        write_one_sentence_to_file(dataset, "train", base_path, "train.txt", "sentence", "label", "_!_")
        write_one_sentence_to_file(dataset, "test", base_path, "test.txt", "sentence", "label", "_!_")
        write_one_sentence_to_file(dataset, "validation", base_path, "val.txt", "sentence", "label", "_!_")
    elif dataset_name == "mnli":
        # write_two_sentence_to_file(dataset, "train", base_path, "poisoning_train.txt", "premise", "hypothesis", "label",
        #                            "_!_", True, 0.1, 1, 2)
        write_two_sentence_to_file(dataset, "train", base_path, "train.txt", "premise", "hypothesis", "label", "_!_")
        write_two_sentence_to_file(dataset, "test_matched", base_path, "test.txt", "premise", "hypothesis", "label",
                                   "_!_")
        write_two_sentence_to_file(dataset, "validation_matched", base_path, "val.txt", "premise", "hypothesis",
                                   "label", "_!_")
    elif dataset_name == "mrpc":
        # write_two_sentence_to_file(dataset, "train", base_path, "poisoning_train.txt", "sentence1", "sentence2",
        #                            "label", "_!_", True, 0.1)
        write_two_sentence_to_file(dataset, "train", base_path, "train.txt", "sentence1", "sentence2", "label", "_!_")
        write_two_sentence_to_file(dataset, "test", base_path, "test.txt", "sentence1", "sentence2", "label", "_!_")
        write_two_sentence_to_file(dataset, "validation", base_path, "val.txt", "sentence1", "sentence2", "label",
                                   "_!_")


def write_one_sentence_to_file(dataset, branch, base_path, extra_path, sentence, label, split, poisoning=False,
                               poisoning_rate=0.1, min=1, max=1):
    if poisoning:
        dataset_length = len(dataset[branch])
        poisoning_index = random.sample(range(dataset_length), k=int(dataset_length * poisoning_rate))
        mod = max + 1
        # print(poisoning_index)
        with open(base_path + extra_path, 'w') as file:
            for index, data in enumerate(dataset[branch]):
                if index in poisoning_index:
                    file.write(data[sentence] + split + str((data[label] + random.randint(min, max)) % mod) + "\n")
                else:
                    file.write(data[sentence] + split + str(data[label]) + "\n")
    else:
        with open(base_path + extra_path, 'w') as file:
            for data in dataset[branch]:
                file.write(data[sentence] + split + str(data[label]) + "\n")


def write_two_sentence_to_file(dataset, branch, base_path, extra_path, premise, hypothesis, label, split,
                               poisoning=False, poisoning_rate=0.1, min=1, max=1):
    if poisoning:
        dataset_length = len(dataset[branch])
        poisoning_index = random.sample(range(dataset_length), k=int(dataset_length * poisoning_rate))
        mod = max + 1
        with open(base_path + extra_path, 'w') as file:
            for index, data in enumerate(dataset[branch]):
                if index in poisoning_index:
                    file.write(data[premise] + split + data[hypothesis] + split + str(
                        (data[label] + random.randint(min, max)) % mod) + "\n")
                else:
                    file.write(data[premise] + split + data[hypothesis] + split + str(data[label]) + "\n")
    else:
        with open(base_path + extra_path, 'w') as file:
            for data in dataset[branch]:
                file.write(data[premise] + split + data[hypothesis] + split + str(data[label]) + "\n")


if __name__ == '__main__':
    # get_imdb_dataset()
    get_glue_dataset()
