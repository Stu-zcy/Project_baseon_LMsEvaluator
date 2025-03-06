import os
import sys
import time
import torch
import logging
from copy import deepcopy
from attack import attack_factory
from tasks.BaseTask import BaseTask
from transformers import BertTokenizer
from utils import LoadChineseNERDataset
from utils.my_exception import print_red
from utils.model_config import ModelTaskForChineseNER
from model.DownstreamTasks import BertForTokenClassification
from sklearn.metrics import accuracy_score, classification_report

sys.path.append('../')



class TaskForChineseNER(BaseTask):
    def __init__(self, initTime, dataset_dir, model_dir,  dataset_type=".txt", use_gpu=True, split_sep=' ',
                 config_parser=None,username='default'):
        self.config = ModelTaskForChineseNER(initTime, dataset_dir, model_dir, dataset_type, use_gpu,
                                             split_sep, config_parser,
                                             username=username)

    def accuracy(self, logits, y_true, ignore_idx=-100):
        """
        :param logits:  [src_len,batch_size,num_labels]
        :param y_true:  [src_len,batch_size]
        :param ignore_idx: 默认情况为-100
        :return:
        e.g.
        y_true = torch.tensor([[-100, 0, 0, 1, -100],
                           [-100, 2, 0, -100, -100]]).transpose(0, 1)
        logits = torch.tensor([[[0.5, 0.1, 0.2], [0.5, 0.4, 0.1], [0.7, 0.2, 0.3], [0.5, 0.7, 0.2], [0.1, 0.2, 0.5]],
                               [[0.3, 0.2, 0.5], [0.7, 0.2, 0.4], [0.8, 0.1, 0.3], [0.9, 0.2, 0.1], [0.1, 0.5, 0.2]]])
        logits = logits.transpose(0, 1)
        print(accuracy(logits, y_true, -100)) # (0.8, 4, 5)
        """
        y_pred = logits.transpose(0, 1).argmax(axis=2).reshape(-1).tolist()
        # 将 [src_len,batch_size,num_labels] 转成 [batch_size, src_len,num_labels]
        y_true = y_true.transpose(0, 1).reshape(-1).tolist()
        real_pred, real_true = [], []
        for item in zip(y_pred, y_true):
            if item[1] != ignore_idx:
                real_pred.append(item[0])
                real_true.append(item[1])
        return accuracy_score(real_true, real_pred), real_true, real_pred

    def train(self):
        model = BertForTokenClassification(self.config,
                                           self.config.pretrained_model_dir)
        model_save_path = os.path.join(self.config.model_save_dir,
                                       self.config.model_save_name)
        global_steps = 0
        if os.path.exists(model_save_path):
            checkpoint = torch.load(model_save_path)
            global_steps = checkpoint['last_epoch']
            loaded_paras = checkpoint['model_state_dict']
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行追加训练......")

        data_loader = LoadChineseNERDataset(
            entities=self.config.entities,
            num_labels=self.config.num_labels,
            ignore_idx=self.config.ignore_idx,
            vocab_path=self.config.vocab_path,
            tokenizer=BertTokenizer.from_pretrained(
                self.config.pretrained_model_dir).tokenize,
            batch_size=self.config.batch_size,
            max_sen_len=self.config.max_sen_len,
            split_sep=self.config.split_sep,
            max_position_embeddings=self.config.max_position_embeddings,
            pad_index=self.config.pad_token_id,
            is_sample_shuffle=self.config.is_sample_shuffle)
        train_iter, test_iter, val_iter = \
            data_loader.load_train_val_test_data(train_file_path=self.config.train_file_path,
                                                 val_file_path=self.config.val_file_path,
                                                 test_file_path=self.config.test_file_path,
                                                 only_test=False)
        model = model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        model.train()
        max_acc = 0
        gradients = []
        for epoch in range(self.config.epochs):
            losses = 0
            start_time = time.time()
            for idx, (sen, token_ids, labels) in enumerate(train_iter):
                token_ids = token_ids.to(self.config.device)
                labels = labels.to(self.config.device)
                padding_mask = (token_ids == data_loader.PAD_IDX).transpose(0, 1)
                loss, logits = model(input_ids=token_ids,  # [src_len, batch_size]
                                     attention_mask=padding_mask,  # [batch_size,src_len]
                                     token_type_ids=None,
                                     position_ids=None,
                                     labels=labels)  # [src_len, batch_size]
                if idx == 0:
                    gradients.append(
                        torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True,
                                            retain_graph=True))
                # logit: [src_len, batch_size, num_labels]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                global_steps += 1
                acc, _, _ = self.accuracy(logits, labels, self.config.ignore_idx)
                if idx % 20 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train acc: {round(acc, 5)}")
                    self.config.writer.add_scalar('Training/Loss', loss.item(), global_steps)
                    self.config.writer.add_scalar('Training/Acc', acc, global_steps)
                if idx % 100 == 0:
                    self.show_result(sen[:10], logits[:, :10], token_ids[:, :10], self.config.entities)
            end_time = time.time()
            train_loss = losses / len(train_iter)
            logging.info(f"Epoch: [{epoch + 1}/{self.config.epochs}],"
                         f" Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            if (epoch + 1) % self.config.model_val_per_epoch == 0:
                acc = self.evaluate(val_iter, model, data_loader)
                logging.info(f"Accuracy on val {acc:.3f}")
                self.config.writer.add_scalar('Testing/Acc', acc, global_steps)
                if acc > max_acc:
                    max_acc = acc
                    state_dict = deepcopy(model.state_dict())
                    torch.save({'last_epoch': global_steps,
                                'model_state_dict': state_dict},
                               model_save_path)
        self.attack(model, BertTokenizer.from_pretrained(self.config.pretrained_model_dir), train_iter,
                    "cos", gradients, data_loader)

    def evaluate(self, val_iter, model, data_loader):
        model.eval()
        real_true, real_pred = [], []
        show = True
        with torch.no_grad():
            for idx, (sen, token_ids, labels) in enumerate(val_iter):
                token_ids = token_ids.to(self.config.device)
                labels = labels.to(self.config.device)
                padding_mask = (token_ids == data_loader.PAD_IDX).transpose(0, 1)
                logits = model(input_ids=token_ids,  # [src_len, batch_size]
                               attention_mask=padding_mask,  # [batch_size,src_len]
                               token_type_ids=None,
                               position_ids=None,
                               labels=None)  # [src_len, batch_size]
                # logits :[src_len, batch_size, num_labels]
                if show:
                    self.show_result(sen[:10], logits[:, :10], token_ids[:, :10], self.config.entities)
                    show = False
                _, t, p = self.accuracy(logits, labels, self.config.ignore_idx)
                real_true += t
                real_pred += p
        model.train()
        target_names = list(self.config.entities.keys())
        logging.info(f"\n{classification_report(real_true, real_pred, target_names=target_names)}")
        return accuracy_score(real_true, real_pred)

    def get_ner_tags(self, logits, token_ids, entities, SEP_IDX=102):
        """
        :param logits:  [src_len,batch_size,num_samples]
        :param token_ids: # [src_len,batch_size]
        :return:
        e.g.
        logits = torch.tensor([[[0.4, 0.7, 0.2],[0.5, 0.4, 0.1],[0.1, 0.2, 0.3],[0.5, 0.7, 0.2],[0.1, 0.2, 0.5]],
                           [[0.3, 0.2, 0.5],[0.7, 0.8, 0.4],[0.1, 0.1, 0.3],[0.9, 0.2, 0.1],[0.1, 0.5,0.2]]])
        logits = logits.transpose(0, 1)  # [src_len,batch_size,num_samples]
        token_ids = torch.tensor([[101, 2769, 511, 102, 0],
                                  [101, 56, 33, 22, 102]]).transpose(0, 1)  # [src_len,batch_size]
        labels, probs = get_ner_tags(logits, token_ids, entities)
        [['O', 'B-LOC'], ['B-ORG', 'B-LOC', 'O']]
        [[0.5, 0.30000001192092896], [0.800000011920929, 0.30000001192092896, 0.8999999761581421]]
        """
        # entities = {'O': 0, 'B-ORG': 1, 'B-LOC': 2, 'B-PER': 3, 'I-ORG': 4, 'I-LOC': 5, 'I-PER': 6}
        label_list = list(entities.keys())
        logits = logits[1:].transpose(0, 1)  # [batch_size,src_len-1,num_samples]
        prob, y_pred = torch.max(logits, dim=-1)  # prob, y_pred: [batch_size,src_len-1]
        token_ids = token_ids[1:].transpose(0, 1)  # [ batch_size,src_len-1]， 去掉[cls]
        assert y_pred.shape == token_ids.shape
        labels = []
        probs = []
        for sample in zip(y_pred, token_ids, prob):
            tmp_label, tmp_prob = [], []
            for item in zip(*sample):
                if item[1] == SEP_IDX:  # 忽略最后一个[SEP]字符
                    break
                tmp_label.append(label_list[item[0]])
                tmp_prob.append(item[2].item())
            labels.append(tmp_label)
            probs.append(tmp_prob)
        return labels, probs

    def pretty_print(self, sentences, labels, entities):
        """
        :param sentences:
        :param labels:
        :param entities:
        :return:
        e.g.
        labels = [['B-PER','I-PER', 'O','O','O','O','O','O','O','O','O','O','B-LOC','I-LOC','B-LOC','I-LOC','O','O','O','O'],
        ['B-LOC','I-LOC','O','B-LOC','I-LOC','O','B-LOC','I-LOC','I-LOC','O','B-LOC','I-LOC','O','O','O','B-PER','I-PER','O','O','O','O','O','O']]
        sentences=["涂伊说，如果有机会他想去赤壁看一看！",
                   "丽江、大理、九寨沟、黄龙等都是涂伊想去的地方！"]
        entities = {'O': 0, 'B-ORG': 1, 'B-LOC': 2, 'B-PER': 3, 'I-ORG': 4, 'I-LOC': 5, 'I-PER': 6}


        句子：涂伊说，如果有机会他想去黄州赤壁看一看！
        涂伊:  PER
        黄州:  LOC
        赤壁:  LOC
        句子：丽江、大理、九寨沟、黄龙等都是涂伊想去的地方！
        丽江:  LOC
        大理:  LOC
        九寨沟:  LOC
        黄龙:  LOC
        涂伊:  PER
        """

        sep_tag = [tag for tag in list(entities.keys()) if 'I' not in tag]
        result = []
        for sen, label in zip(sentences, labels):
            logging.info(f"句子：{sen}")
            last_tag = None
            for item in zip(sen + "O", label + ['O']):
                if item[1] in sep_tag:  #
                    if len(result) > 0:
                        entity = "".join(result)
                        logging.info(f"\t{entity}:  {last_tag.split('-')[-1]}")
                        result = []
                    if item[1] != 'O':
                        result.append(item[0])
                        last_tag = item[1]
                else:
                    result.append(item[0])
                    last_tag = item[1]

    def show_result(self, sentences, logits, token_ids, entities):
        labels, _ = self.get_ner_tags(logits, token_ids, entities)
        self.pretty_print(sentences, labels, entities)

    def inference(self, sentences=None):
        model = BertForTokenClassification(self.config,
                                           self.config.pretrained_model_dir)
        model_save_path = os.path.join(self.config.model_save_dir,
                                       self.config.model_save_name)
        if os.path.exists(model_save_path):
            checkpoint = torch.load(model_save_path)
            loaded_paras = checkpoint['model_state_dict']
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行追加训练......")
        else:
            raise ValueError(f" 本地模型{model_save_path}不存在，请先训练模型。")
        model = model.to(self.config.device)
        data_loader = LoadChineseNERDataset(
            entities=self.config.entities,
            num_labels=self.config.num_labels,
            ignore_idx=self.config.ignore_idx,
            vocab_path=self.config.vocab_path,
            tokenizer=BertTokenizer.from_pretrained(
                self.config.pretrained_model_dir).tokenize,
            batch_size=self.config.batch_size,
            max_sen_len=self.config.max_sen_len,
            split_sep=self.config.split_sep,
            max_position_embeddings=self.config.max_position_embeddings,
            pad_index=self.config.pad_token_id,
            is_sample_shuffle=self.config.is_sample_shuffle)
        _, token_ids, _ = data_loader.make_inference_samples(sentences)
        token_ids = token_ids.to(self.config.device)
        padding_mask = (token_ids == data_loader.PAD_IDX).transpose(0, 1)
        logits = model(input_ids=token_ids,  # [src_len, batch_size]
                       attention_mask=padding_mask)  # [batch_size,src_len]
        self.show_result(sentences, logits, token_ids, self.config.entities)

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
        sentences = ['智光拿出石壁拓文为乔峰详述事情始末，乔峰方知自己原本姓萧，乃契丹后族。',
                     '当乔峰问及带头大哥时，却发现智光大师已圆寂。',
                     '乔峰、阿朱相约找最后知情人康敏问完此事后，就到塞外骑马牧羊，再不回来。']
        self.inference(sentences)




if __name__ == '__main__':
    task = TaskForChineseNER("ChineseNER", "bert_base_chinese", "example_", ".txt", True, ' ')
    task.run()
