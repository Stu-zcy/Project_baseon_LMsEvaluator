import os
import sys
import time
import torch
import logging
import threading   # --- 新增 ---
from tqdm import tqdm

# 根据实际项目目录结构，import 可能需要调整
from attack import attack_helper
from tasks.BaseTask import BaseTask
from transformers import BertTokenizer
from utils.my_exception import print_red
from utils.my_prettytable import MyPrettyTable
from utils import LoadSingleSentenceClassificationDataset
from model.DownstreamTasks import BertForSentenceClassification
from utils.model_config import ModelTaskForSingleSentenceClassification

sys.path.append('..')


class TaskForSingleSentenceClassification(BaseTask):
    def __init__(self, dataset_dir, model_dir, dataset_type=".txt", use_gpu=True, split_sep='_!_',
                 config_parser=None, username='default'):
        self.config = ModelTaskForSingleSentenceClassification(
            dataset_dir, model_dir, dataset_type,
            use_gpu, split_sep, config_parser, username=username
        )
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
        self.result = {}

        # --- 新增：初始化进度字典 ---
        self.attack_progress = {
            "Task": "TaskForSingleSentenceClassification",
            "progress": 0,      # 0~100
            "status": "idle",   # train | inference | attack | finished ...
            "message": ""
        }

        # --- 新增：保存线程引用 ---
        self.thread = None

    # --- 新增：提供一个获取进度的公共方法 ---
    def get_progress(self):
        return self.attack_progress

    def run_in_thread(self):
        """
        在一个新线程中异步执行 run() 以避免阻塞主线程。
        """
        # 如果有一个线程已经在跑，且还没结束，可以在这里做出保护
        if self.thread and self.thread.is_alive():
            logging.warning("任务线程已经在运行中，无法再次启动。")
            return

        # 创建并启动线程
        pipeline_thread = threading.Thread(target=self.run)
        pipeline_thread.start()
        self.thread = pipeline_thread

    def train(self):
        self.attack_progress["status"] = "train"
        self.attack_progress["progress"] = 0
        self.attack_progress["message"] = "Training started..."

        model = BertForSentenceClassification(self.config, self.config.pretrained_model_dir)
        model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')

        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path, map_location=self.config.device)
            model.load_state_dict(loaded_paras)
            logging.info("## 已有模型存储路径: " + str(model_save_path))
            logging.info("## 成功载入已有模型，进行追加训练......")

        model = model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        model.train()

        bert_tokenize = BertTokenizer.from_pretrained(self.config.pretrained_model_dir).tokenize
        data_loader = LoadSingleSentenceClassificationDataset(
            vocab_path=self.config.vocab_path,
            tokenizer=bert_tokenize,
            batch_size=self.config.batch_size,
            max_sen_len=self.config.max_sen_len,
            split_sep=self.config.split_sep,
            max_position_embeddings=self.config.max_position_embeddings,
            pad_index=self.config.pad_token_id,
            is_sample_shuffle=self.config.is_sample_shuffle
        )
        train_iter, test_iter, val_iter, sequence_iter = data_loader.load_train_val_test_data(
            self.config.train_file_path,
            self.config.val_file_path,
            self.config.test_file_path
        )
        self.data_loader = data_loader
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter

        max_acc = 0
        sample_losses, gradients = [], []

        num_epochs = self.config.epochs
        for epoch in range(num_epochs):
            losses = 0
            start_time = time.time()
            pbar = tqdm(train_iter)
            total_batches = len(train_iter)

            for idx, (sample, label) in enumerate(pbar):
                if idx == 1000:  # 仅示例：只跑前1000个batch
                    break

                sample = sample.to(self.config.device)
                label = label.to(self.config.device)
                padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)

                loss, logits = model(
                    input_ids=sample,
                    attention_mask=padding_mask,
                    token_type_ids=None,
                    position_ids=None,
                    labels=label
                )
                sample_losses.append(loss)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                if idx == 0:
                    gradients.append(
                        torch.autograd.grad(
                            loss, model.parameters(),
                            create_graph=False, allow_unused=True,
                            retain_graph=True
                        )
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                acc = (logits.argmax(1) == label).float().mean()
                pbar.set_description(
                    f"Epoch: {epoch}, Batch[{idx}/{total_batches}], "
                    f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}"
                )

                # --- 实时更新进度 ---
                current_batch_progress = (idx + 1) / total_batches
                epoch_progress = (epoch + current_batch_progress) / num_epochs
                self.attack_progress["progress"] = round(epoch_progress * 100, 2)
                self.attack_progress["message"] = (
                    f"Training Epoch {epoch+1}/{num_epochs}, "
                    f"Batch {idx+1}/{total_batches}, "
                    f"Loss: {loss.item():.4f}, Acc: {acc:.3f}"
                )

            end_time = time.time()
            train_loss = losses / len(train_iter)
            logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
                         f"Epoch time = {(end_time - start_time):.3f}s")

            acc = self.evaluate(val_iter, model, self.config.device, data_loader.PAD_IDX)
            logging.info(f"Accuracy on val {acc:.3f}")
            if acc > max_acc:
                max_acc = acc
                if epoch != 0:
                    logging.info("update max_acc: " + str(acc))
                    logging.info("## 进行已有模型的更新，模型路径：" + str(model_save_path))
                    torch.save(model.state_dict(), model_save_path)

        self.model = model
        self.attack_train_iter = train_iter
        self.attack_gradients = gradients
        self.attack_data_loader = data_loader
        self.attack_optimizer = optimizer
        self.attack_bert_tokenize = bert_tokenize

        # --- 训练结束 ---
        self.attack_progress["progress"] = 100
        self.attack_progress["message"] = "Training finished!"
        logging.info("## Training finished")

    def inference(self):
        self.attack_progress["status"] = "inference"
        self.attack_progress["progress"] = 0
        self.attack_progress["message"] = "Inference started..."

        model = self.model
        model_save_path = os.path.join(self.config.model_save_dir, 'model.pt')

        if os.path.exists(model_save_path):
            loaded_paras = torch.load(model_save_path, map_location=self.config.device)
            model.load_state_dict(loaded_paras)
            logging.info("## 已有模型存储路径: " + str(model_save_path))
            logging.info("## 成功载入已有模型，进行预测......")

        model = model.to(self.config.device)
        if self.data_loader is None or self.test_iter is None:
            data_loader = LoadSingleSentenceClassificationDataset(
                vocab_path=self.config.vocab_path,
                tokenizer=BertTokenizer.from_pretrained(self.config.pretrained_model_dir).tokenize,
                batch_size=self.config.batch_size,
                max_sen_len=self.config.max_sen_len,
                split_sep=self.config.split_sep,
                max_position_embeddings=self.config.max_position_embeddings,
                pad_index=self.config.pad_token_id,
                is_sample_shuffle=self.config.is_sample_shuffle
            )
            train_iter, test_iter, val_iter, _ = data_loader.load_train_val_test_data(
                self.config.train_file_path,
                self.config.val_file_path,
                self.config.test_file_path
            )
            self.data_loader = data_loader
            self.train_iter = train_iter
            self.test_iter = test_iter
            self.val_iter = val_iter

        acc = self.evaluate(self.test_iter, model, device=self.config.device,
                            PAD_IDX=self.data_loader.PAD_IDX)
        logging.info(f"Acc on test:{acc:.3f}")
        self.table_show(acc)

        self.attack_progress["progress"] = 100
        self.attack_progress["message"] = f"Inference finished! Test acc: {acc:.3f}"

        logging.info("=" * 50)
        logging.info("模型预测结束")
        logging.info("=" * 50)

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
            for x, y in data_iter:
                x, y = x.to(device), y.to(device)
                padding_mask = (x == PAD_IDX).transpose(0, 1)
                logits = model(x, attention_mask=padding_mask)
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
                break  # 示例：只跑1个 batch
            model.train()
            return acc_sum / n if n != 0 else 0

    def attack(self, distance_func):
        self.attack_progress["status"] = "attack"
        self.attack_progress["progress"] = 0
        self.attack_progress["message"] = "Attack started..."

        if len(self.config.config_parser['attack_list']) == 0:
            logging.info("=" * 50)
            logging.info("没有攻击被执行。")
            logging.info("=" * 50)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_model_dir)
            total_attacks = len(self.config.config_parser['attack_list'])
            for index in range(total_attacks):
                attack_config = self.config.config_parser['attack_list'][index]['attack_args']
                attack_type = attack_config['attack_type']
                attack_model = None
                logging.info("=" * 50)
                logging.info("攻击模块配置检查")

                if attack_type == "GIAforNLP":
                    attack_model = attack_helper.AttackHelper(
                        attack_type,
                        self.config.config_parser,
                        attack_config,
                        model=self.model,
                        tokenizer=tokenizer,
                        train_iter=self.attack_train_iter,
                        distance_func=distance_func,
                        gradients=self.attack_gradients,
                        data_loader=self.attack_data_loader,
                    )
                elif (attack_type in ["AdvAttack", "SWAT", "BackDoorAttack"]):
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
                        task_config=self.config,
                        model=self.model,
                        optimizer=self.attack_optimizer,
                        bert_tokenize=self.attack_bert_tokenize,
                    )
                else:
                    print_red("UNKNOWN ATTACK TYPE CONFIG.")
                    print_red("Please check the attackArgs.attackType config in config.yaml.")

                logging.info(attack_type + " 攻击开始")
                res = attack_model.attack()
                attack_list = self.result.setdefault(attack_model.attack_type, [])
                attack_list.append(res)
                logging.info(attack_type + " 攻击结束")
                logging.info("=" * 50)

                # 更新攻击进度（逐个攻击执行，假设依次完成）
                self.attack_progress["progress"] = round(((index + 1) / total_attacks) * 100, 2)
                self.attack_progress["message"] = (
                    f"Attack {attack_type} finished, index {index+1}/{total_attacks}"
                )

        self.attack_progress["message"] = "All attacks finished."
        logging.info("攻击流程已全部结束")

    def run(self):
        self.train()
        self.inference()
        self.attack("cos")
        self.attack_progress["status"] = "finished"
        self.attack_progress["message"] = "All pipeline finished!"
        self.attack_progress["progress"] = 100


if __name__ == "__main__":
    task = TaskForSingleSentenceClassification(
        "SingleSentenceClassification",
        "bert_base_chinese",
        ".txt", True, '_!_'
    )
    # --- 改为使用线程启动 ---
    task.run_in_thread()

    # 也可以在此处循环查看进度，仅示例
    while task.thread.is_alive():
        print("Progress:", task.get_progress())
        time.sleep(2)

    print("最终进度:", task.get_progress())
