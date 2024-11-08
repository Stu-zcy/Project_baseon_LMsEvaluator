import os
import sys
import time
import torch
import logging
from copy import deepcopy
from attack import attack_helper
from torch.optim.adamw import AdamW
from tasks.BaseTask import BaseTask
from transformers import BertTokenizer
from utils.my_exception import print_red
from utils import LoadBertPretrainingDataset
from utils.model_config import ModelTaskForPretraining
from model.DownstreamTasks import BertForPretrainingModel
from transformers import get_polynomial_decay_schedule_with_warmup

sys.path.append('../')


class TaskForPretraining(BaseTask):
    def __init__(self, dataset_dir, model_dir, dataset_type=".txt", use_gpu=True,
                 data_name="songci", config_parser=None):
        self.config = ModelTaskForPretraining(dataset_dir, model_dir, dataset_type,
                                              use_gpu, data_name, config_parser)

    def get_model(self):
        model = BertForPretrainingModel(self.config,
                                        self.config.pretrained_model_dir)
        return model, BertTokenizer.from_pretrained(self.config.pretrained_model_dir)

    def train(self):
        model = BertForPretrainingModel(self.config,
                                        self.config.pretrained_model_dir)
        last_epoch = -1
        if os.path.exists(self.config.model_save_path):
            checkpoint = torch.load(self.config.model_save_path)
            last_epoch = checkpoint['last_epoch']
            loaded_paras = checkpoint['model_state_dict']
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型，进行追加训练......")
        model = model.to(self.config.device)
        model.train()
        bert_tokenize = BertTokenizer.from_pretrained(self.config.pretrained_model_dir).tokenize
        data_loader = LoadBertPretrainingDataset(vocab_path=self.config.vocab_path,
                                                 tokenizer=bert_tokenize,
                                                 batch_size=self.config.batch_size,
                                                 max_sen_len=self.config.max_sen_len,
                                                 max_position_embeddings=self.config.max_position_embeddings,
                                                 pad_index=self.config.pad_index,
                                                 is_sample_shuffle=self.config.is_sample_shuffle,
                                                 random_state=self.config.random_state,
                                                 data_name=self.config.data_name,
                                                 masked_rate=self.config.masked_rate,
                                                 masked_token_rate=self.config.masked_token_rate,
                                                 masked_token_unchanged_rate=self.config.masked_token_unchanged_rate)
        train_iter, test_iter, val_iter = \
            data_loader.load_train_val_test_data(test_file_path=self.config.test_file_path,
                                                 train_file_path=self.config.train_file_path,
                                                 val_file_path=self.config.val_file_path)
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
                "initial_lr": self.config.learning_rate

            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "initial_lr": self.config.learning_rate
            },
        ]
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              int(len(train_iter) * 0),
                                                              int(self.config.epochs * len(train_iter)),
                                                              last_epoch=last_epoch)
        max_acc = 0
        state_dict = None
        gradients = []
        for epoch in range(self.config.epochs):
            losses = 0
            start_time = time.time()
            for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(train_iter):
                b_token_ids = b_token_ids.to(self.config.device)  # [src_len, batch_size]
                b_segs = b_segs.to(self.config.device)
                b_mask = b_mask.to(self.config.device)
                b_mlm_label = b_mlm_label.to(self.config.device)
                b_nsp_label = b_nsp_label.to(self.config.device)
                loss, mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                                     attention_mask=b_mask,
                                                     token_type_ids=b_segs,
                                                     masked_lm_labels=b_mlm_label,
                                                     next_sentence_labels=b_nsp_label)
                if idx == 0:
                    gradients.append(
                        torch.autograd.grad(loss, model.parameters(), create_graph=False, allow_unused=True,
                                            retain_graph=True))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses += loss.item()
                mlm_acc, _, _, nsp_acc, _, _ = self.accuracy(mlm_logits, nsp_logits, b_mlm_label,
                                                             b_nsp_label, data_loader.PAD_IDX)
                if idx % 20 == 0:
                    logging.info(f"Epoch: [{epoch + 1}/{self.config.epochs}], Batch[{idx}/{len(train_iter)}], "
                                 f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f},"
                                 f"nsp acc: {nsp_acc:.3f}")
                    self.config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                    self.config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0],
                                                  scheduler.last_epoch)
                    self.config.writer.add_scalars(main_tag='Training/Accuracy',
                                                   tag_scalar_dict={'NSP': nsp_acc,
                                                                    'MLM': mlm_acc},
                                                   global_step=scheduler.last_epoch)
            end_time = time.time()
            train_loss = losses / len(train_iter)
            logging.info(f"Epoch: [{epoch + 1}/{self.config.epochs}], Train loss: "
                         f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            if (epoch + 1) % self.config.model_val_per_epoch == 0:
                mlm_acc, nsp_acc = self.evaluate(val_iter, model, data_loader.PAD_IDX)
                logging.info(f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, "
                             f"NSP Accuracy on val: {round(nsp_acc, 4)}")
                self.config.writer.add_scalars(main_tag='Testing/Accuracy',
                                               tag_scalar_dict={'NSP': nsp_acc,
                                                                'MLM': mlm_acc},
                                               global_step=scheduler.last_epoch)
                # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
                if mlm_acc > max_acc:
                    max_acc = mlm_acc
                    state_dict = deepcopy(model.state_dict())
                torch.save({'last_epoch': scheduler.last_epoch,
                            'model_state_dict': state_dict},
                           self.config.model_save_path)
        self.attack(model, BertTokenizer.from_pretrained(self.config.pretrained_model_dir), train_iter,
                    "cos", gradients, data_loader)

    def accuracy(self, mlm_logits, nsp_logits, mlm_labels, nsp_label, PAD_IDX):
        """
        :param mlm_logits:  [src_len,batch_size,src_vocab_size]
        :param mlm_labels:  [src_len,batch_size]
        :param nsp_logits:  [batch_size,2]
        :param nsp_label:  [batch_size]
        :param PAD_IDX:
        :return:
        """
        mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
        # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
        mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
        # 将 [src_len,batch_size] 转成 [batch_size， src_len]
        mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况，得到预测正确的个数（此时还包括有mask位置）
        mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
        mlm_acc = mlm_acc.logical_and(mask)  # 去掉mlm_acc中mask的部分
        mlm_correct = mlm_acc.sum().item()
        mlm_total = mask.sum().item()
        mlm_acc = float(mlm_correct) / mlm_total

        nsp_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
        nsp_total = len(nsp_label)
        nsp_acc = float(nsp_correct) / nsp_total
        return [mlm_acc, mlm_correct, mlm_total, nsp_acc, nsp_correct, nsp_total]

    def evaluate(self, data_iter, model, PAD_IDX):
        model.eval()
        mlm_corrects, mlm_totals, nsp_corrects, nsp_totals = 0, 0, 0, 0
        with torch.no_grad():
            for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(data_iter):
                b_token_ids = b_token_ids.to(self.config.device)  # [src_len, batch_size]
                b_segs = b_segs.to(self.config.device)
                b_mask = b_mask.to(self.config.device)
                b_mlm_label = b_mlm_label.to(self.config.device)
                b_nsp_label = b_nsp_label.to(self.config.device)
                mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                               attention_mask=b_mask,
                                               token_type_ids=b_segs)
                result = self.accuracy(mlm_logits, nsp_logits, b_mlm_label, b_nsp_label, PAD_IDX)
                _, mlm_cor, mlm_tot, _, nsp_cor, nsp_tot = result
                mlm_corrects += mlm_cor
                mlm_totals += mlm_tot
                nsp_corrects += nsp_cor
                nsp_totals += nsp_tot
        model.train()
        return [float(mlm_corrects) / mlm_totals, float(nsp_corrects) / nsp_totals]

    def inference(self, sentences=None, masked=False, language='en', random_state=None):
        """
        :param config:
        :param sentences:
        :param masked: 推理时的句子是否Mask
        :param language: 语种
        :param random_state:  控制mask字符时的随机状态
        :return:
        """
        bert_tokenize = BertTokenizer.from_pretrained(self.config.pretrained_model_dir).tokenize
        data_loader = LoadBertPretrainingDataset(vocab_path=self.config.vocab_path,
                                                 tokenizer=bert_tokenize,
                                                 pad_index=self.config.pad_index,
                                                 random_state=self.config.random_state,
                                                 masked_rate=0.15)  # 15% Mask掉
        token_ids, pred_idx, mask = data_loader.make_inference_samples(sentences,
                                                                       masked=masked,
                                                                       language=language,
                                                                       random_state=random_state)
        model = BertForPretrainingModel(self.config,
                                        self.config.pretrained_model_dir)
        if os.path.exists(self.config.model_save_path):
            checkpoint = torch.load(self.config.model_save_path)
            loaded_paras = checkpoint['model_state_dict']
            model.load_state_dict(loaded_paras)
            logging.info("## 成功载入已有模型进行推理......")
        else:
            raise ValueError(f"模型 {self.config.model_save_path} 不存在！")
        model = model.to(self.config.device)
        model.eval()
        with torch.no_grad():
            token_ids = token_ids.to(self.config.device)  # [src_len, batch_size]
            mask = mask.to(self.config.device)
            mlm_logits, _ = model(input_ids=token_ids,
                                  attention_mask=mask)
        self.pretty_print(token_ids, mlm_logits, pred_idx,
                          data_loader.vocab.itos, sentences, language)

    def pretty_print(self, token_ids, logits, pred_idx, itos, sentences, language):
        """
        格式化输出结果
        :param token_ids:   [src_len, batch_size]
        :param logits:  [src_len, batch_size, vocab_size]
        :param pred_idx:   二维列表，每个内层列表记录了原始句子中被mask的位置
        :param itos:
        :param sentences: 原始句子
        :return:
        """
        token_ids = token_ids.transpose(0, 1)  # [batch_size,src_len]
        logits = logits.transpose(0, 1)  # [batch_size, src_len,vocab_size]
        y_pred = logits.argmax(axis=2)  # [batch_size, src_len]
        sep = " " if language == 'en' else ""
        for token_id, sentence, y, y_idx in zip(token_ids, sentences, y_pred, pred_idx):
            sen = [itos[id] for id in token_id]
            sen_mask = sep.join(sen).replace(" ##", "").replace("[PAD]", "").replace(" ,", ",")
            sen_mask = sen_mask.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
            logging.info(f" ### 原始: {sentence}")
            logging.info(f"  ## 掩盖: {sen_mask}")
            for idx in y_idx:
                sen[idx] = itos[y[idx]].replace("##", "")
            sen = sep.join(sen).replace("[PAD]", "").replace(" ,", ",")
            sen = sen.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
            logging.info(f"  ## 预测: {sen}")
            logging.info("=" * 50)

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
        sentences_1 = ["I no longer love her, true, but perhaps I love her.",
                       "Love is so short and oblivion so long."]
        sentences_2 = ["十年生死两茫茫。不思量。自难忘。千里孤坟，无处话凄凉。",
                       "红酥手。黄藤酒。满园春色宫墙柳。"]
        self.inference(sentences_2, masked=False, language='zh', random_state=2022)


if __name__ == '__main__':
    task = TaskForPretraining("SongCi", "bert_base_chinese", "songci_", ".txt", True, "songci")
    task.run()
