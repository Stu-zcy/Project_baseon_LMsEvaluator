import os
import sys
import time
import torch
import random
import logging
from attack.base_attack import BaseAttack
from utils.my_prettytable import MyPrettyTable
from transformers import BertTokenizer, get_scheduler
from model.DownstreamTasks import BertForSentenceClassification
from utils import LoadSingleSentenceClassificationDataset, LoadPairSentenceClassificationDataset


class PoisoningAttack(BaseAttack):
    def __init__(self, config_parser, attack_config, poisoning_rate=0.1, epochs=10, task_config=None, model=None,
                 optimizer=None, bert_tokenize=None, display_full_info=False):
        super().__init__(config_parser, attack_config)
        self.poisoning_rate = poisoning_rate
        self.model = model
        self.config = task_config
        self.optimizer = optimizer
        self.bert_tokenize = bert_tokenize
        self.config.epochs = epochs
        self.display_full_info = display_full_info
        self.my_handlers = logging.getLogger().handlers
        self.__get_poisoning_data_path()

    def attack(self):
        if self.config_parser['task_config']['task'] == "TaskForSingleSentenceClassification":
            return self.attack_single_sentence_classification()
        elif self.config_parser['task_config']['task'] == "TaskForPairSentenceClassification":
            return self.attack_pair_sentence_classification()

    def attack_single_sentence_classification(self):
        model_save_path = os.path.join(self.config.model_save_dir, 'poisoning_model.pt')
        data_loader = LoadSingleSentenceClassificationDataset(vocab_path=self.config.vocab_path,
                                                              tokenizer=self.bert_tokenize,
                                                              batch_size=self.config.batch_size,
                                                              max_sen_len=self.config.max_sen_len,
                                                              split_sep=self.config.split_sep,
                                                              max_position_embeddings=self.config.max_position_embeddings,
                                                              pad_index=self.config.pad_token_id,
                                                              is_sample_shuffle=self.config.is_sample_shuffle)
        train_iter, test_iter, val_iter, sequence_iter = data_loader.load_train_val_test_data(
            self.config.train_file_path,
            self.config.val_file_path,
            self.config.test_file_path)
        min_acc = 1
        sample_losses, gradients = [], []
        for epoch in range(self.config.epochs):
            losses = 0
            start_time = time.time()
            for idx, (sample, label) in enumerate(train_iter):
                sample = sample.to(self.config.device)  # [src_len, batch_size]
                label = label.to(self.config.device)
                padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
                loss, logits = self.model(
                    input_ids=sample,
                    attention_mask=padding_mask,
                    token_type_ids=None,
                    position_ids=None,
                    labels=label)
                sample_losses.append(loss)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                if idx == 0:
                    gradients.append(
                        torch.autograd.grad(loss, self.model.parameters(), create_graph=False, allow_unused=True,
                                            retain_graph=True))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses += loss.item()
                acc = (logits.argmax(1) == label).float().mean()
                if idx % 1000 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
            end_time = time.time()
            train_loss = losses / len(train_iter)
            logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            # if (epoch + 1) % self.config.model_val_per_epoch == 0:
            acc = self.evaluate_single_sentence_classification(val_iter, self.model, self.config.device,
                                                               data_loader.PAD_IDX)
            logging.info(f"Accuracy on val {acc:.3f}")
            # torch.save(self.model.state_dict(), model_save_path)
            if acc < min_acc:
                min_acc = acc
                if not epoch == 0:
                    logging.info("## 进行已有模型的更新，更新模型路径为：" + str(model_save_path))
                    torch.save(self.model.state_dict(), model_save_path)
        self.model = BertForSentenceClassification(self.config,
                                                   self.config.pretrained_model_dir)
        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path, map_location=self.config.device)
            self.model.load_state_dict(loaded_paras)
            logging.info("## 已有被攻击后模型存储路径: " + str(model_save_path))
            logging.info("## 成功载入已有被攻击后模型，进行预测......")
        self.model = self.model.to(self.config.device)
        # data_loader = LoadSingleSentenceClassificationDataset(vocab_path=self.config.vocab_path,
        #                                                       tokenizer=BertTokenizer.from_pretrained(
        #                                                           self.config.pretrained_model_dir).tokenize,
        #                                                       batch_size=self.config.batch_size,
        #                                                       max_sen_len=self.config.max_sen_len,
        #                                                       split_sep=self.config.split_sep,
        #                                                       max_position_embeddings=self.config.max_position_embeddings,
        #                                                       pad_index=self.config.pad_token_id,
        #                                                       is_sample_shuffle=self.config.is_sample_shuffle)
        # train_iter, test_iter, val_iter, _ = data_loader.load_train_val_test_data(self.config.train_file_path,
        #                                                                           self.config.val_file_path,
        #                                                                           self.config.test_file_path)
        acc = self.evaluate_single_sentence_classification(test_iter, self.model, device=self.config.device,
                                                           PAD_IDX=data_loader.PAD_IDX)
        logging.info(f"Acc on test:{acc:.3f}")

        logging.info("## 对原始模型性能进行检测")
        model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')
        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path, map_location=self.config.device)
            self.model.load_state_dict(loaded_paras)
            logging.info("## 已有干净模型存储路径: " + str(model_save_path))
            logging.info("## 成功载入干净模型，进行预测......")
        self.model = self.model.to(self.config.device)
        ori_acc = self.evaluate_single_sentence_classification(test_iter, self.model, device=self.config.device,
                                                               PAD_IDX=data_loader.PAD_IDX)
        self.__table_show(acc, ori_acc)
        return [ori_acc, acc]

    def attack_pair_sentence_classification(self):
        # model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')
        # if os.path.exists(model_save_path):
        #     loaded_paras = torch.load(model_save_path, map_location=self.config.device)
        #     self.model.load_state_dict(loaded_paras)
        #     logging.info("## 成功载入已有模型，进行追加训练......")
        # self.model = self.model.to(self.config.device)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # self.model.train()
        # bert_tokenize = BertTokenizer.from_pretrained(
        #     self.config.pretrained_model_dir).tokenize
        model_save_path = os.path.join(self.config.model_save_dir, 'poisoning_model.pt')
        data_loader = LoadPairSentenceClassificationDataset(
            vocab_path=self.config.vocab_path,
            tokenizer=self.bert_tokenize,
            batch_size=self.config.batch_size,
            max_sen_len=self.config.max_sen_len,
            split_sep=self.config.split_sep,
            max_position_embeddings=self.config.max_position_embeddings,
            pad_index=self.config.pad_token_id)
        train_iter, test_iter, val_iter = \
            data_loader.load_train_val_test_data(self.config.train_file_path,
                                                 self.config.val_file_path,
                                                 self.config.test_file_path)
        lr_scheduler = get_scheduler(name='linear',
                                     optimizer=self.optimizer,
                                     num_warmup_steps=int(len(train_iter) * 0),
                                     num_training_steps=int(self.config.epochs * len(train_iter)))
        min_acc = 1
        gradients = []
        for epoch in range(self.config.epochs):
            losses = 0
            start_time = time.time()
            for idx, (sample, seg, label) in enumerate(train_iter):
                sample = sample.to(self.config.device)  # [src_len, batch_size]
                label = label.to(self.config.device)
                seg = seg.to(self.config.device)
                padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
                loss, logits = self.model(
                    input_ids=sample,
                    attention_mask=padding_mask,
                    token_type_ids=seg,
                    position_ids=None,
                    labels=label)
                if idx == 0:
                    gradients.append(
                        torch.autograd.grad(loss, self.model.parameters(), create_graph=False, allow_unused=True,
                                            retain_graph=True))
                self.optimizer.zero_grad()
                loss.backward()
                lr_scheduler.step()
                self.optimizer.step()
                losses += loss.item()
                acc = (logits.argmax(1) == label).float().mean()
                if idx % 1000 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
            end_time = time.time()
            train_loss = losses / len(train_iter)
            logging.info(f"Epoch: {epoch}, Train loss: "
                         f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            if (epoch + 1) % self.config.model_val_per_epoch == 0:
                acc = self.evaluate_pair_sentence_classification(val_iter, self.model, self.config.device,
                                                                 data_loader.PAD_IDX)
                logging.info(f"Accuracy on val {acc:.3f}")
                # torch.save(self.model.state_dict(), model_save_path)
                if acc < min_acc:
                    min_acc = acc
                    if not epoch == 0:
                        logging.info("## 进行已有模型的更新，更新模型路径为：" + str(model_save_path))
                        torch.save(self.model.state_dict(), model_save_path)
        self.model = BertForSentenceClassification(self.config,
                                                   self.config.pretrained_model_dir)
        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path, map_location=self.config.device)
            self.model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行预测......")
        self.model = self.model.to(self.config.device)
        data_loader = LoadPairSentenceClassificationDataset(vocab_path=self.config.vocab_path,
                                                            tokenizer=BertTokenizer.from_pretrained(
                                                                self.config.pretrained_model_dir).tokenize,
                                                            batch_size=self.config.batch_size,
                                                            max_sen_len=self.config.max_sen_len,
                                                            split_sep=self.config.split_sep,
                                                            max_position_embeddings=self.config.max_position_embeddings,
                                                            pad_index=self.config.pad_token_id,
                                                            is_sample_shuffle=self.config.is_sample_shuffle)
        test_iter = data_loader.load_train_val_test_data(test_file_path=self.config.test_file_path,
                                                         only_test=True)
        acc = self.evaluate(test_iter, self.model, device=self.config.device, PAD_IDX=data_loader.PAD_IDX)
        logging.info(f"Acc on test:{acc:.3f}")

        logging.info("## 对原始模型性能进行检测")
        model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')
        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path, map_location=self.config.device)
            self.model.load_state_dict(loaded_paras)
            logging.info("## 已有干净模型存储路径: " + str(model_save_path))
            logging.info("## 成功载入干净模型，进行预测......")
        self.model = self.model.to(self.config.device)
        ori_acc = self.evaluate(test_iter, self.model, device=self.config.device, PAD_IDX=data_loader.PAD_IDX)
        self.__table_show(acc, ori_acc)
        return [ori_acc, acc]

    def evaluate_single_sentence_classification(self, data_iter, model, device, PAD_IDX):
        model.eval()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                padding_mask = (x == PAD_IDX).transpose(0, 1)
                logits = model(x, attention_mask=padding_mask)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
            model.train()
            return acc_sum / n

    def __table_show(self, acc, ori_acc=0.5):
        my_pretty_table = MyPrettyTable()
        my_pretty_table.add_field_names(['Results', ''])
        my_pretty_table.add_row(['Original accuracy', f"{(ori_acc * 100):.3f}%"])
        my_pretty_table.add_row(['Accuracy under poisoning attack', f"{(acc * 100):.3f}%"])
        my_pretty_table.print_table()
        my_pretty_table.logging_table()

    def evaluate_pair_sentence_classification(self, data_iter, model, device, PAD_IDX):
        model.eval()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, seg, y in data_iter:
                x, seg, y = x.to(device), seg.to(device), y.to(device)
                padding_mask = (x == PAD_IDX).transpose(0, 1)
                logits = model(x, attention_mask=padding_mask, token_type_ids=seg)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
            model.train()
            return acc_sum / n

    def __get_poisoning_data_path(self):
        poisoning_train_path, file_type = self.__deal_path(self.config.train_file_path)
        self.__make_poisoning_data(self.config.train_file_path, poisoning_train_path)
        self.config.train_file_path = poisoning_train_path

    def __deal_path(self, ori_path):
        try:
            base_path, file_name = ori_path.rsplit('/', 1)
        except:
            base_path, file_name = ori_path.rsplit('\\', 1)
        file_name = "poisoning_" + file_name
        result = base_path + "/" + file_name
        return result, "." + file_name.split('.')[-1]

    def __make_poisoning_data(self, ori_path, poisoning_path):
        logging.info("生在生成投毒数据")
        logging.info("原始数据集路径: " + ori_path)
        logging.info("投毒数据集路径: " + poisoning_path)
        with open(ori_path, "r", encoding="utf-8") as file:
            str_list = file.readlines()
            input_list, label_list = [], []
            label_set = set()
            length_data = len(str_list)
            for str in str_list:
                input, label = str[:-1].rsplit('_!_', 1)
                input_list.append(input)
                label_list.append(label)
                label_set.add(label)

        poisoning_index = random.sample(range(length_data), k=int(length_data * self.poisoning_rate))
        label_size = len(label_set)
        label_set = list(label_set)
        for index in poisoning_index:
            ori_index = label_set.index(label_list[index])
            poisoning_index = (ori_index + random.randint(1, label_size - 1)) % label_size
            label_list[index] = label_set[poisoning_index]

        with open(poisoning_path, "w", encoding="utf-8") as file:
            for input, label in zip(input_list, label_list):
                file.writelines(input + "_!_" + label + "\n")
        logging.info("投毒数据生成完毕")


if __name__ == "__main__":
    # import random

    # temp = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], k=8)
    # print(temp)
    # temp = random.randint(1, 1)
    # print(temp)

    poisoning_rate = 0.1
    with open("../../datasets/imdb/train.txt", "r", encoding="utf-8") as file:
        str_list = file.readlines()
        input_list, label_list = [], []
        label_set = set()
        length_data = len(str_list)
        for str in str_list:
            input, label = str[:-1].rsplit('_!_', 1)
            input_list.append(input)
            label_list.append(label)
            label_set.add(label)

    poisoning_index = random.sample(range(length_data), k=int(length_data * poisoning_rate))
    label_size = len(label_set)
    label_set = list(label_set)
    for index in poisoning_index:
        ori_index = label_set.index(label_list[index])
        poisoning_index = (ori_index + random.randint(1, label_size - 1)) % label_size
        label_list[index] = label_set[poisoning_index]

    with open("../../datasets/imdb/poisoning_train.txt", "w", encoding="utf-8") as file:
        for input, label in zip(input_list, label_list):
            file.writelines(input + "_!_" + label + "\n")
