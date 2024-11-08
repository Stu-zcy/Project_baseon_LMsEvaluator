import time
import torch
import logging
import evaluate
import attack.GIAforNLP.GIA_attack_helper
import numpy as np
from utils.my_exception import print_red

BERT_CLS_TOKEN = 101
BERT_SEP_TOKEN = 102
BERT_PAD_TOKEN = 0


class MyGIAforNLP(attack.GIAforNLP.GIA_attack_helper.AttackModel):
    def __init__(self, attack_config, model, tokenizer, train_iter, distance_func, gradients, config_parser,
                 data_loader):
        super().__init__(model, tokenizer, train_iter, distance_func, gradients,
                         config_parser)
        self.attack_config = attack_config
        self.data_loader = data_loader

    def attack(self):
        attack_nums = self.config_parser['attack_args']['attack_nums']
        for i in range(attack_nums):
            print(f"第{i + 1}次攻击\n")
            batch_size = self.config_parser['attack_args']['attack_batch']
            sample, label = self.get_sample(batch_size)
            gradient = self.gradients[0]
            # sys.exit()
            my_input, runtime = self.reconstruct(sample, label, gradient)

            unused_tokens = self.get_unused_tokens(gradient)
            _, best_ids = self.get_closest_tokens(my_input, unused_tokens, batch_size,
                                                  metric=self.config_parser['attack_args']['distance_func'])

            prediction = []
            for i in range(best_ids.shape[0]):
                prediction.append(self.remove_padding(best_ids[i]))
            logging.info("优化后预测结果：")
            logging.info(prediction)

            prediction, reference = [], []
            my_ids, ori_ids = [], []
            for i in range(batch_size):
                my_ids.append(my_input[:, i].tolist())
                ori_ids.append(sample[:, i].tolist())
            for my_id, ori_id in zip(my_ids, ori_ids):
                prediction.append(self.tokenizer.decode(my_id))
                reference.append(self.tokenizer.decode(ori_id))
            logging.info("没有优化的预测结果：")
            logging.info(prediction)
            logging.info("原始输入：")
            logging.info(reference)

            # # 分数评估(evaluate.load("rouge"))需要挂梯子
            # score = self.compute_scores(prediction, reference, evaluate.load("rouge"))
            # print("重构得分：")
            # print(score)

    def get_sample(self, batch_size):
        sample, label = None, None
        sample_list, label_list = [], []
        max_len = 0
        for idx, (samples, labels) in enumerate(self.train_iter):
            if idx == batch_size:
                break
            if samples.shape[0] > max_len:
                max_len = samples.shape[0]

            if samples.shape[1] == 1:
                temp_sample = samples.to(self.device)
                temp_label = labels.to(self.device)
            else:
                temp_sample = samples[:, 0].to(self.device)
                temp_label = labels[0].to(self.device)

            sample_list.append(temp_sample)
            label_list.append(temp_label)

        for sample_ori, label_ori in zip(sample_list, label_list):
            temp_sample = torch.cat((sample_ori, torch.zeros([max_len - sample_ori.shape[0], 1]).to(self.device)),
                                    dim=0)
            if sample is None:
                sample = temp_sample
                label = label_ori.reshape(1, 1)
            else:
                sample = torch.cat((sample, temp_sample), dim=1)
                label = torch.cat((label, label_ori.reshape(1, 1)), dim=1)
        return sample, label

    def reconstruct(self, sample, label, gradient):
        # 预测向量初始化
        my_input = torch.randint(0, 21129, sample.shape).to(self.device)

        # 优化器设置
        if self.config_parser['attack_args']['optimizer'] == 'Adam':
            opt = torch.optim.Adam([my_input], lr=self.config_parser['attack_args']['attack_lr'])
        elif self.config_parser['attack_args']['optimizer'] == 'LBFGS':
            opt = torch.optim.LBFGS([my_input], lr=self.config_parser['attack_args']['attack_lr'])
        else:
            print_red("Please check the attack_args.optimizer config in config.yaml.")
            print_red("Using Adam optimizer.")
            opt = torch.optim.Adam([my_input], lr=self.config_parser['attack_args']['attack_lr'])

        # 迭代更新
        best_final_error, best_final_x = None, my_input.detach().clone()
        time_start = time.time()
        for it in range(self.config_parser['attack_args']['attack_iters']):
            def closure():
                padding_mask = (my_input == self.data_loader.PAD_IDX).transpose(0, 1)
                my_loss, my_logits = self.model(
                    input_ids=my_input,
                    attention_mask=padding_mask,
                    token_type_ids=None,
                    position_ids=None,
                    labels=label)
                opt.zero_grad()
                my_gradient = torch.autograd.grad(my_loss, self.model.parameters(), create_graph=False,
                                                  allow_unused=True, retain_graph=True)
                distance = self.distance_func(my_gradient, gradient)
                distance.requires_grad_(True)
                distance.backward(retain_graph=True)
                return distance

            error = opt.step(closure)
            if (it + 1) % 10 == 0:
                print(f"第{it + 1}次迭代：当前误差为{error}，用时为{time.time() - time_start}秒")
            if best_final_error is None or error <= best_final_error:
                best_final_error = error.item()
                best_final_x = my_input
            del error
        runtime = time.time() - time_start
        return my_input.to(self.device), runtime

    def get_unused_tokens(self, true_grads):
        unused_tokens = []
        for i in range(self.tokenizer.vocab_size):
            if true_grads[0][i].abs().sum() < 1e-9 and i != BERT_PAD_TOKEN:
                unused_tokens += [i]
        unused_tokens = np.array(unused_tokens)
        return unused_tokens

    def compute_scores(self, prediction, reference, metric):
        rouge = metric.compute(predictions=prediction, references=reference)
        return rouge

    def get_closest_tokens(self, input_ids, unused_tokens, batch_size, metric='cos'):
        ori_inputs_embeds = self.model.bert.bert_embeddings(input_ids=input_ids,
                                                            token_type_ids=None,
                                                            position_ids=None)
        inputs_embeds = None
        list = []
        for temp in ori_inputs_embeds:
            if len(list) == 0:
                for i in range(batch_size):
                    list.append(temp[i, :].reshape(1, 768))
            else:
                for i in range(batch_size):
                    list[i] = torch.cat((list[i], temp[i, :].reshape(1, 768)), dim=0)
        shape = list[0].shape
        for i in range(batch_size):
            if inputs_embeds is None:
                inputs_embeds = list[i].reshape(1, shape[0], shape[1])
            else:
                inputs_embeds = torch.cat((inputs_embeds, list[i].reshape(1, shape[0], shape[1])), dim=0)

        bert_embeddings_weight = self.model.bert.bert_embeddings.word_embeddings.embedding.weight.unsqueeze(0)
        embeddings_weight = bert_embeddings_weight.repeat(self.config_parser['attack_args']['attack_batch'], 1, 1)
        inputs_embeds = inputs_embeds.view(batch_size, len(ori_inputs_embeds), 768)
        if metric == 'l2':
            d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
        elif metric == 'cos':
            dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
            norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
            norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
            d = -dp / (norm1 * norm2)
        else:
            assert False

        d[:, :, unused_tokens] = 1e9
        return d, d.min(dim=2)[1]

    def init_get_closest_tokens(self, inputs_embeds, unused_tokens, model, metric='cos'):
        bert_embeddings = model.get_input_embeddings()
        bert_embeddings_weight = bert_embeddings.weight.unsqueeze(0)
        embeddings_weight = bert_embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
        if metric == 'l2':
            d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
        elif metric == 'cos':
            dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
            norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
            norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
            d = -dp / (norm1 * norm2)
        else:
            assert False

        d[:, :, unused_tokens] = 1e9
        return d, d.min(dim=2)[1]

    def remove_padding(self, ids):
        for i in range(ids.shape[0] - 1, -1, -1):
            if ids[i] == BERT_SEP_TOKEN:
                ids = ids[:i + 1]
                break
        return self.tokenizer.decode(ids)
