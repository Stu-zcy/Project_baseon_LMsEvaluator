import os
import sys
import time
import torch
import logging
import numpy as np
from attack import attack_factory
from tasks.BaseTask import BaseTask
from transformers import BertTokenizer
from utils.my_exception import print_red
from utils import LoadMultipleChoiceDataset
from model.DownstreamTasks import BertForMultipleChoice
from utils.model_config import ModelTaskForMultipleChoice

sys.path.append('../')


class TaskForMultipleChoice(BaseTask):
    def __init__(self, initTime, dataset_dir, model_dir,  dataset_type=".csv", use_gpu=True, config_parser=None,username='default'):
        self.config = ModelTaskForMultipleChoice(initTime, dataset_dir, model_dir, dataset_type,
                                                 use_gpu, config_parser,
																								 username=username)

    def train(self):
        model = BertForMultipleChoice(self.config,
                                      self.config.pretrained_model_dir)
        model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')
        model_load_path = os.path.join(self.config.model_load_dir, 'model.pt')
        if os.path.exists(model_load_path):
            loaded_paras = torch.load(model_load_path)
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行追加训练......")
        model = model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        model.train()
        bert_tokenize = BertTokenizer.from_pretrained(self.config.pretrained_model_dir).tokenize
        data_loader = LoadMultipleChoiceDataset(
            vocab_path=self.config.vocab_path,
            tokenizer=bert_tokenize,
            batch_size=self.config.batch_size,
            max_sen_len=self.config.max_sen_len,
            max_position_embeddings=self.config.max_position_embeddings,
            pad_index=self.config.pad_token_id,
            is_sample_shuffle=self.config.is_sample_shuffle,
            num_choice=self.config.num_labels)
        train_iter, test_iter, val_iter = \
            data_loader.load_train_val_test_data(self.config.train_file_path,
                                                 self.config.val_file_path,
                                                 self.config.test_file_path)
        max_acc = 0
        gradients = []
        for epoch in range(self.config.epochs):
            losses = 0
            start_time = time.time()
            for idx, (qa, seg, mask, label) in enumerate(train_iter):
                qa = qa.to(self.config.device)  # [src_len, batch_size]
                label = label.to(self.config.device)
                seg = seg.to(self.config.device)
                mask = mask.to(self.config.device)
                loss, logits = model(input_ids=qa,
                                     attention_mask=mask,
                                     token_type_ids=seg,
                                     position_ids=None,
                                     labels=label)
                if idx == 0:
                    gradients.append(
                        torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True,
                                            retain_graph=True))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                acc = (logits.argmax(1) == label).float().mean()
                if idx % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
                if idx % 100 == 0:
                    y_pred = logits.argmax(1).cpu()
                    self.show_result(qa, y_pred, data_loader.vocab.itos, num_show=1)
            end_time = time.time()
            train_loss = losses / len(train_iter)
            logging.info(f"Epoch: {epoch}, Train loss: "
                         f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            if (epoch + 1) % self.config.model_val_per_epoch == 0:
                acc, _ = self.evaluate(val_iter, model,
                                       self.config.device, inference=False)
                logging.info(f"Accuracy on val {acc:.3f}")
                if acc > max_acc:
                    max_acc = acc
                    torch.save(model.state_dict(), model_save_path)
        self.attack(model, BertTokenizer.from_pretrained(self.config.pretrained_model_dir), train_iter,
                    "cos", gradients, data_loader)

    def inference(self):
        model = BertForMultipleChoice(self.config,
                                      self.config.pretrained_model_dir)
        model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')
        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path)
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行预测......")
        model = model.to(self.config.device)
        data_loader = LoadMultipleChoiceDataset(vocab_path=self.config.vocab_path,
                                                tokenizer=BertTokenizer.from_pretrained(
                                                    self.config.pretrained_model_dir).tokenize,
                                                batch_size=self.config.batch_size,
                                                max_sen_len=self.config.max_sen_len,
                                                max_position_embeddings=self.config.max_position_embeddings,
                                                pad_index=self.config.pad_token_id,
                                                is_sample_shuffle=self.config.is_sample_shuffle)
        test_iter = data_loader.load_train_val_test_data(test_file_path=self.config.test_file_path,
                                                         only_test=True)
        y_pred = self.evaluate(test_iter, model, self.config.device, inference=True)
        logging.info(f"预测标签为：{y_pred.tolist()}")

    def evaluate(self, data_iter, model, device, inference=False):
        model.eval()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            y_pred = []
            for qa, seg, mask, y in data_iter:
                qa, seg, y, mask = qa.to(device), seg.to(device), y.to(device), mask.to(device)
                logits = model(qa, attention_mask=mask, token_type_ids=seg)
                y_pred.append(logits.argmax(1).cpu().numpy())
                if not inference:
                    acc_sum += (logits.argmax(1) == y).float().sum().item()
                    n += len(y)
            model.train()
            if inference:
                return np.hstack(y_pred)
            return acc_sum / n, np.hstack(y_pred)

    def show_result(self, qas, y_pred, itos=None, num_show=5):
        count = 0
        num_samples, num_choice, seq_len = qas.size()
        qas = qas.reshape(-1)
        strs = np.array([itos[t] for t in qas]).reshape(-1, seq_len)
        for i in range(num_samples):  # 遍历每个样本
            s_idx = i * num_choice
            e_idx = s_idx + num_choice
            sample = strs[s_idx:e_idx]
            if count == num_show:
                return
            count += 1
            for j, item in enumerate(sample):  # 每个样本的四个答案
                q, a, _ = " ".join(item[1:]).replace(" .", ".").replace(" ##", "").split('[SEP]')
                if y_pred[i] == j:
                    a += " ## True"
                else:
                    a += " ## False"
                logging.info(f"[{num_show}/{count}] ### {q + a}")
            logging.info("\n")

    def attack(self, model, tokenizer, train_iter, distance_func, gradients, data_loader):
        if len(self.config.config_parser['attack_list']) == 0:
            logging.info("=" * 50)
            logging.info("没有攻击被执行。")
            logging.info("=" * 50)
        else:
            logging.info("此任务不适用于攻击方法的测试")
            return

    def run(self):
        self.train()
        self.inference()


if __name__ == '__main__':
    task = TaskForMultipleChoice("MultipleChoice", "bert_base_uncased_english", "", ".csv", True)
    task.run()
