import os
import sys
import time
import torch
import logging
from tqdm import tqdm
from attack import attack_factory
from tasks.BaseTask import BaseTask
from utils.my_prettytable import MyPrettyTable
from transformers import BertTokenizer, get_scheduler
from utils import LoadPairSentenceClassificationDataset
from utils.model_config import ModelTaskForPairSentenceClassification
from model.DownstreamTasks import BertForPretrainingModel, BertForSentenceClassification

sys.path.append('../')


class TaskForPairSentenceClassification(BaseTask):
    def __init__(self, initTime, dataset_dir, model_dir,  dataset_type=".txt", use_gpu=True, split_sep='_!_',
                 config_parser=None,username='default'):
        self.config = ModelTaskForPairSentenceClassification(initTime, dataset_dir, model_dir, dataset_type,
                                                             use_gpu, split_sep, config_parser,username=username)
        self.model = None
        self.data_loader = None
        self.train_iter = None
        self.test_iter = None
        self.val_iter = None
        self.attack_train_iter = None
        self.attack_gradients = None
        self.attack_data_loader = None
        self.attack_optimizer = None
        self.attack_bert_tokenize = None

    def train(self):
        model = BertForSentenceClassification(self.config,
                                              self.config.pretrained_model_dir)
        model_save_path = os.path.join(self.config.model_save_dir, 'task4mrpc_model.pt')
        model_load_path = os.path.join(self.config.model_load_dir, 'task4mrpc_model.pt')
        min_loss, max_acc = float('inf'), 0
        if os.path.exists(model_load_path):
            loaded_paras = torch.load(model_load_path)
            loaded_paras = torch.load(model_load_path, map_location=self.config.device, weights_only=True)
            model.load_state_dict(loaded_paras['model_state_dict'])
            min_loss = loaded_paras['min_loss']
            max_acc = loaded_paras['max_acc']
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
        sample_losses, gradients, len_iter = [], [], len(train_iter)
        for epoch in range(self.config.epochs):
            total_loss, total_acc = 0, 0
            start_time = time.time()
            pbar = tqdm(train_iter)
            for idx, (sample, seg, label) in enumerate(pbar):
                current_len = idx + 1
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
                total_loss += loss.item()
                acc = (logits.argmax(1) == label).float().mean()
                total_acc += acc.item()
                pbar.set_description(
                    f"Epoch: {epoch}, Batch[{idx}/{len_iter}]")
                pbar.set_postfix(
                    {"Avg Loss": f"{total_loss / current_len:.3f}",
                     "Avg Train Acc": f"{total_acc / current_len:.3f}"}
                )
                if idx % 10 == 0:
                    logging.info(f"Epoch: {epoch}, Batch[{idx}/{len_iter}], "
                                 f"Avg loss :{loss.item():.3f}, Avg acc: {acc:.3f}")
                break
            end_time = time.time()
            train_loss = total_loss / len_iter
            logging.info(f"Epoch: {epoch}, Train loss: "
                         f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
            if (epoch + 1) % self.config.model_val_per_epoch == 0:
                acc = self.evaluate(val_iter, model, self.config.device, data_loader.PAD_IDX)
                logging.info(f"Accuracy on val {acc:.3f}")
                if acc > max_acc:
                    max_acc = acc
                    min_loss = total_loss
                    logging.info("update max_acc: " + str(acc))
                    logging.info(f"## epoch: {epoch}, min_loss: {min_loss:.3f}, max_loss: {max_acc:.3f}")
                    logging.info("## 进行已有模型的更新，更新模型路径为：" + str(model_save_path))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'min_loss': train_loss,
                        'max_acc': max_acc
                    }, model_save_path)
        self.model = model
        self.attack_train_iter = train_iter
        self.attack_gradients = gradients
        self.attack_data_loader = data_loader
        self.attack_optimizer = optimizer
        self.attack_bert_tokenize = bert_tokenize

    def temp_save(self, model, model_save_path):
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        torch.save({
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'min_loss': 1,
            'max_acc': 0
        }, model_save_path)

    def load_model(self, model_save_path=None):
        '''默认读取为预训练后模型，默认路径"/cache/modelOutput/task4mrpc_model.pt"'''
        model = BertForSentenceClassification(self.config, self.config.pretrained_model_dir)

        if model_save_path is None:
            model_save_path = os.path.join(self.config.model_save_dir, 'task4sst_model.pt')
            self.temp_save(model, model_save_path)
        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path, map_location=self.config.device, weights_only=True)
            model.load_state_dict(loaded_paras['model_state_dict'])
            logging.info("## 已有模型存储路径: " + str(model_save_path))
            logging.info("## 成功载入已有模型，进行预测......")
        self.model = model.to(self.config.device)

    def inference(self):
        # model = BertForSentenceClassification(self.config,
        #                                       self.config.pretrained_model_dir)
        self.load_model()
        model = self.model.to(self.config.device)
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
            pbar = tqdm(data_iter)
            len_iter = len(data_iter)
            for idx, (x, seg, y) in enumerate(data_iter):
                x, seg, y = x.to(device), seg.to(device), y.to(device)
                padding_mask = (x == PAD_IDX).transpose(0, 1)
                logits = model(x, attention_mask=padding_mask, token_type_ids=seg)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
                pbar.set_description(f"Batch[{idx}/{len_iter}]")
                pbar.set_postfix({"Avg Acc": f"{(acc_sum / n):.3f}"})
                break
            model.train()
            return acc_sum / n

    def attack(self):
        if len(self.config.config_parser['attack_list']) == 0:
            logging.info("=" * 50)
            logging.info("没有攻击被执行。")
            logging.info("=" * 50)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model_dir)
            self.load_model()
            for index in range(len(self.config.config_parser['attack_list'])):
                attack_config = self.config.config_parser['attack_list'][index]['attack_args']
                attack_type = attack_config['attack_type']
                attack_mode = None
                logging.info("=" * 50)
                logging.info("攻击模块配置检查")
                if attack_type == "GIAforNLP":
                    attack_mode = attack_factory.AttackFactory(
                        attack_type,
                        self.config.config_parser,
                        attack_config,
                        self.config.device,
                        model=self.model,
                        tokenizer=tokenizer,
                        train_iter=self.attack_train_iter,
                        gradients=self.attack_gradients,
                        data_loader=self.attack_data_loader,
                    )
                elif attack_type == "PoisoningAttack":
                    attack_mode = attack_factory.AttackFactory(
                        attack_type,
                        self.config.config_parser,
                        attack_config,
                        self.config.device,
                        task_config=self.config,
                        model=self.model,
                        optimizer=self.attack_optimizer,
                        bert_tokenize=self.attack_bert_tokenize,
                    )
                else:
                    attack_mode = attack_factory.AttackFactory(
                        attack_type,
                        self.config.config_parser,
                        attack_config,
                        self.config.device,
                    )
                logging.info(attack_type + " 攻击开始")
                attack_mode.attack()
                logging.info(attack_type + "攻击结束")
                logging.info("=" * 50)

    def run(self):
        # self.train()
        # self.inference()
        self.attack()


if __name__ == '__main__':
    task = TaskForPairSentenceClassification("PairSentenceClassification", "bert_base_uncased", ".txt", True,
                                             '_!_')
    task.run()

