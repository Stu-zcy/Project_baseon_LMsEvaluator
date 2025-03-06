import os
import sys
import time
import torch
import logging
import collections

from attack import attack_factory
from tasks.BaseTask import BaseTask
from utils.my_exception import print_red
from attack.GIAforNLP import my_GIA_for_NLP
from transformers import BertTokenizer, get_scheduler
from model.DownstreamTasks import BertForQuestionAnswering
from utils.data_helpers import LoadSQuADQuestionAnsweringDataset
from utils.model_config import ModelTaskForSQuADQuestionAnswering

sys.path.append('../')


class TaskForSQuADQuestionAnswering(BaseTask):
    def __init__(self, initTime, dataset_dir, model_dir,  dataset_type=".json", use_gpu=True, config_parser=None,username='default'):
        self.config = ModelTaskForSQuADQuestionAnswering(initTime, dataset_dir, model_dir, dataset_type, use_gpu,
                                                         config_parser, username=username)

    def train(self):
        model = BertForQuestionAnswering(self.config,
                                         self.config.pretrained_model_dir)
        if os.path.exists(self.config.model_save_path):
            loaded_paras = torch.load(self.config.model_save_path)
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行追加训练......")
        model = model.to(self.config.device)

        model.train()
        bert_tokenize = BertTokenizer.from_pretrained(self.config.pretrained_model_dir).tokenize
        data_loader = LoadSQuADQuestionAnsweringDataset(vocab_path=self.config.vocab_path,
                                                        tokenizer=bert_tokenize,
                                                        batch_size=self.config.batch_size,
                                                        max_sen_len=self.config.max_sen_len,
                                                        max_query_length=self.config.max_query_len,
                                                        max_position_embeddings=self.config.max_position_embeddings,
                                                        pad_index=self.config.pad_token_id,
                                                        is_sample_shuffle=self.config.is_sample_shuffle,
                                                        doc_stride=self.config.doc_stride)
        train_iter, test_iter, val_iter = \
            data_loader.load_train_val_test_data(train_file_path=self.config.train_file_path,
                                                 test_file_path=self.config.test_file_path,
                                                 only_test=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_scheduler(name='linear',
                                     optimizer=optimizer,
                                     num_warmup_steps=int(len(train_iter) * 0),
                                     num_training_steps=int(self.config.epochs * len(train_iter)))
        max_acc = 0
        gradients = []
        for epoch in range(self.config.epochs):
            losses = 0
            start_time = time.time()
            for idx, (batch_input, batch_seg, batch_label, _, _, _, _) in enumerate(train_iter):
                batch_input = batch_input.to(self.config.device)  # [src_len, batch_size]
                batch_seg = batch_seg.to(self.config.device)
                batch_label = batch_label.to(self.config.device)
                padding_mask = (batch_input == data_loader.PAD_IDX).transpose(0, 1)
                loss, start_logits, end_logits = model(input_ids=batch_input,
                                                       attention_mask=padding_mask,
                                                       token_type_ids=batch_seg,
                                                       position_ids=None,
                                                       start_positions=batch_label[:, 0],
                                                       end_positions=batch_label[:, 1])
                if idx == 0:
                    gradients.append(
                        torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True,
                                            retain_graph=True))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                losses += loss.item()
                acc_start = (start_logits.argmax(1) == batch_label[:, 0]).float().mean()
                acc_end = (end_logits.argmax(1) == batch_label[:, 1]).float().mean()
                acc = (acc_start + acc_end) / 2
                if idx % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
                if idx % 100 == 0:
                    y_pred = [start_logits.argmax(1), end_logits.argmax(1)]
                    y_true = [batch_label[:, 0], batch_label[:, 1]]
                    self.show_result(batch_input, data_loader.vocab.itos,
                                     y_pred=y_pred, y_true=y_true)
            end_time = time.time()
            train_loss = losses / len(train_iter)
            logging.info(f"Epoch: {epoch}, Train loss: "
                         f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            if (epoch + 1) % self.config.model_val_per_epoch == 0:
                acc = self.evaluate(val_iter, model,
                                    self.config.device,
                                    data_loader.PAD_IDX,
                                    inference=False)
                logging.info(f" ### Accuracy on val: {round(acc, 4)} max :{max_acc}")
                if acc > max_acc:
                    max_acc = acc
                torch.save(model.state_dict(), self.config.model_save_path)
        self.attack(model, BertTokenizer.from_pretrained(self.config.pretrained_model_dir), train_iter,
                    "cos", gradients, data_loader)

    def evaluate(self, data_iter, model, device, PAD_IDX, inference=False):
        model.eval()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            all_results = collections.defaultdict(list)
            for batch_input, batch_seg, batch_label, batch_qid, _, batch_feature_id, _ in data_iter:
                batch_input = batch_input.to(device)  # [src_len, batch_size]
                batch_seg = batch_seg.to(device)
                batch_label = batch_label.to(device)
                padding_mask = (batch_input == PAD_IDX).transpose(0, 1)
                start_logits, end_logits = model(input_ids=batch_input,
                                                 attention_mask=padding_mask,
                                                 token_type_ids=batch_seg,
                                                 position_ids=None)
                # 将同一个问题下的所有预测样本的结果保存到一个list中，这里只对batchsize=1时有用
                all_results[batch_qid[0]].append([batch_feature_id[0],
                                                  start_logits.cpu().numpy().reshape(-1),
                                                  end_logits.cpu().numpy().reshape(-1)])
                if not inference:
                    acc_sum_start = (start_logits.argmax(1) == batch_label[:, 0]).float().sum().item()
                    acc_sum_end = (end_logits.argmax(1) == batch_label[:, 1]).float().sum().item()
                    acc_sum += (acc_sum_start + acc_sum_end)
                    n += len(batch_label)
            model.train()
            if inference:
                return all_results
            return acc_sum / (2 * n)

    def show_result(self, batch_input, itos, num_show=5, y_pred=None, y_true=None):
        """
        本函数的作用是在训练模型的过程中展示相应的结果
        :param batch_input:
        :param itos:
        :param num_show:
        :param y_pred:
        :param y_true:
        :return:
        """
        count = 0
        batch_input = batch_input.transpose(0, 1)  # 转换为[batch_size, seq_len]形状
        for i in range(len(batch_input)):  # 取一个batch所有的原始文本
            if count == num_show:
                break
            input_tokens = [itos[s] for s in batch_input[i]]  # 将question+context 的ids序列转为字符串
            start_pos, end_pos = y_pred[0][i], y_pred[1][i]
            answer_text = " ".join(input_tokens[start_pos:(end_pos + 1)]).replace(" ##", "")
            input_text = " ".join(input_tokens).replace(" ##", "").split('[SEP]')
            question_text, context_text = input_text[0], input_text[1]

            logging.info(f"### Question: {question_text}")
            logging.info(f"  ## Predicted answer: {answer_text}")
            start_pos, end_pos = y_true[0][i], y_true[1][i]
            true_answer_text = " ".join(input_tokens[start_pos:(end_pos + 1)])
            true_answer_text = true_answer_text.replace(" ##", "")
            logging.info(f"  ## True answer: {true_answer_text}")
            logging.info(f"  ## True answer idx: {start_pos.cpu(), end_pos.cpu()}")
            count += 1

    def inference(self):
        bert_tokenize = BertTokenizer.from_pretrained(self.config.pretrained_model_dir).tokenize
        data_loader = LoadSQuADQuestionAnsweringDataset(vocab_path=self.config.vocab_path,
                                                        tokenizer=bert_tokenize,
                                                        batch_size=1,  # 只能是1
                                                        max_sen_len=self.config.max_sen_len,
                                                        doc_stride=self.config.doc_stride,
                                                        max_query_length=self.config.max_query_len,
                                                        max_answer_length=self.config.max_answer_len,
                                                        max_position_embeddings=self.config.max_position_embeddings,
                                                        pad_index=self.config.pad_token_id,
                                                        n_best_size=self.config.n_best_size)
        test_iter, all_examples = data_loader.load_train_val_test_data(test_file_path=self.config.test_file_path,
                                                                       only_test=True)
        model = BertForQuestionAnswering(self.config,
                                         self.config.pretrained_model_dir)
        if os.path.exists(self.config.model_save_path):
            loaded_paras = torch.load(self.config.model_save_path)
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，开始进行推理......")
        else:
            raise ValueError(f"## 模型{self.config.model_save_path}不存在，请检查路径或者先训练模型......")

        model = model.to(self.config.device)
        all_result_logits = self.evaluate(test_iter, model, self.config.device,
                                          data_loader.PAD_IDX, inference=True)
        data_loader.write_prediction(test_iter, all_examples,
                                     all_result_logits, self.config.dataset_dir)

    def run(self):
        self.train()
        self.inference()

    def attack(self, model, tokenizer, train_iter, distance_func, gradients, data_loader):
        if len(self.config.config_parser['attack_list']) == 0:
            logging.info("=" * 50)
            logging.info("没有攻击被执行。")
            logging.info("=" * 50)
        else:
            logging.info("此任务不适用于攻击方法的测试")
            return


if __name__ == "__main__":
    task = TaskForSQuADQuestionAnswering("SQuAD", "bert_base_uncased_english", "v1-", ".json", True)
    task.run()
