import os
import sys
import time
import torch
import logging
from tqdm import tqdm
from attack import attack_helper
from tasks.BaseTask import BaseTask
from utils.my_exception import print_red
from utils.my_prettytable import MyPrettyTable
from transformers import BertTokenizer, get_scheduler
from utils import LoadPairSentenceClassificationDataset
from model.DownstreamTasks import BertForSentenceClassification
from utils.model_config import ModelTaskForPairSentenceClassification

sys.path.append('../')


class TaskForPairSentenceClassification(BaseTask):
    def __init__(self, dataset_dir, model_dir, dataset_type=".txt", use_gpu=True, split_sep='_!_',
                 config_parser=None,username='default'):
        self.config = ModelTaskForPairSentenceClassification(dataset_dir, model_dir, dataset_type,
                                                             use_gpu, split_sep, config_parser,username=username)

    def train(self):
        model = BertForSentenceClassification(self.config,
                                              self.config.pretrained_model_dir)
        model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')
        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path)
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行追加训练......")
        model = model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        model.train()
        bert_tokenize = BertTokenizer.from_pretrained(
            self.config.pretrained_model_dir).tokenize
        data_loader = LoadPairSentenceClassificationDataset(
            vocab_path=self.config.vocab_path,
            tokenizer=bert_tokenize,
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
                                     optimizer=optimizer,
                                     num_warmup_steps=int(len(train_iter) * 0),
                                     num_training_steps=int(self.config.epochs * len(train_iter)))
        max_acc = 0
        gradients = []
        for epoch in range(self.config.epochs):
            losses = 0
            start_time = time.time()
            pbar = tqdm(train_iter)
            for idx, (sample, seg, label) in enumerate(pbar):
                sample = sample.to(self.config.device)  # [src_len, batch_size]
                label = label.to(self.config.device)
                seg = seg.to(self.config.device)
                padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
                loss, logits = model(
                    input_ids=sample,
                    attention_mask=padding_mask,
                    token_type_ids=seg,
                    position_ids=None,
                    labels=label)
                if idx == 0:
                    gradients.append(
                        torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True,
                                            retain_graph=True))
                optimizer.zero_grad()
                loss.backward()
                lr_scheduler.step()
                optimizer.step()
                losses += loss.item()
                acc = (logits.argmax(1) == label).float().mean()
                pbar.set_description(
                    f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
                if idx % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")

            end_time = time.time()
            train_loss = losses / len(train_iter)
            logging.info(f"Epoch: {epoch}, Train loss: "
                         f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            if (epoch + 1) % self.config.model_val_per_epoch == 0:
                acc = self.evaluate(val_iter, model, self.config.device, data_loader.PAD_IDX)
                logging.info(f"Accuracy on val {acc:.3f}")
                if acc > max_acc:
                    max_acc = acc
                    torch.save(model.state_dict(), model_save_path)
        self.attack(model, BertTokenizer.from_pretrained(self.config.pretrained_model_dir), train_iter,
                    "cos", gradients, data_loader, self.config, optimizer, bert_tokenize)

    def inference(self):
        model = BertForSentenceClassification(self.config,
                                              self.config.pretrained_model_dir)
        model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')
        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path)
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行预测......")
        model = model.to(self.config.device)
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
        acc = self.evaluate(test_iter, model, device=self.config.device, PAD_IDX=data_loader.PAD_IDX)
        logging.info(f"Acc on test:{acc:.3f}")
        self.table_show(acc)

    def table_show(self, acc):
        acc_format = "{:.3%}".format(acc)
        my_pretty_table = MyPrettyTable()
        my_pretty_table.add_field_names(['Results', ''])
        my_pretty_table.add_row(['Accuracy', acc_format])
        my_pretty_table.print_table()
        my_pretty_table.logging_table()

    def evaluate(self, data_iter, model, device, PAD_IDX):
        model.eval()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, seg, y in data_iter:
                x, seg, y = x.to(device), seg.to(device), y.to(device)
                padding_mask = (x == PAD_IDX).transpose(0, 1)
                logits = model(x, attention_mask=padding_mask, token_type_ids=seg)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
                break
            model.train()
            return acc_sum / n

    def attack(self, model, tokenizer, train_iter, distance_func, gradients, data_loader, task_config, optimizer,
               bert_tokenize):
        if len(self.config.config_parser['attack_list']) == 0:
            logging.info("=" * 50)
            logging.info("没有攻击被执行。")
            logging.info("=" * 20)
        else:
            for index in range(len(self.config.config_parser['attack_list'])):
                attack_config = self.config.config_parser['attack_list'][index]['attack_args']
                attack_type = attack_config['attack_type']
                attack_model = None
                if attack_type == "GIAforNLP":
                    attack_model = attack_helper.AttackHelper(
                        attack_type,
                        self.config.config_parser,
                        attack_config,
                        model=model,
                        tokenizer=tokenizer,
                        train_iter=train_iter,
                        distance_func=distance_func,
                        gradients=gradients,
                        data_loader=data_loader,
                    )
                elif (attack_type == "AdvAttack" or attack_type == "SWAT"
                      or attack_type == "BackDoorAttack"):
                    attack_model = attack_helper.AttackHelper(
                        attack_type,
                        self.config.config_parser,
                        attack_config,
                    )
                elif attack_type == "PoisoningAttack":
                    attack_model = attack_helper.AttackHelper(
                        attack_type,
                        self.config.config_parser,
                        attack_config,
                        task_config=task_config,
                        model=model,
                        optimizer=optimizer,
                        bert_tokenize=bert_tokenize,
                    )
                else:
                    print_red("UNKNOWN ATTACK TYPE CONFIG.")
                    print_red("Please check the attackArgs.attackType config in config.yaml.")
                logging.info("=" * 50)
                logging.info("攻击开始")
                attack_model.attack()
                logging.info("攻击结束")
                logging.info("=" * 50)

    def run(self):
        self.train()
        self.inference()


if __name__ == '__main__':
    task = TaskForPairSentenceClassification("PairSentenceClassification", "bert_base_uncased_english", "",
                                             ".txt", True, '_!_')
    task.run()
