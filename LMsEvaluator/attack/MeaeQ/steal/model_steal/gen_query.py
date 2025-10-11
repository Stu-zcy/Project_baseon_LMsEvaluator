import sys
import torch
import logging
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, XLNetTokenizer

from attack.MeaeQ.utils.config import *
from attack.MeaeQ.utils.tools import *
from attack.MeaeQ.models.victim_models import BFSC, RFSC, XFSC, GPT2FSC

# args = ArgParser().get_parser()
rate_label = 0.0
label_result = ''


class my_gen_query():
    def __init__(self, my_args=None, **kwargs):

        sys.path.append('../../../')
        self.args = ArgParser().get_parser()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.visible_device)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ids = [0, 1]
        logging.info(self.args.victim_model_version)
        # print(self.args.victim_model_version)
        if self.args.victim_model_version == 'bert_base_uncased':
            self.victim_model = BFSC(self.args)
        elif self.args.victim_model_version == 'roberta_base':
            self.victim_model = RFSC(self.args)
        elif self.args.victim_model_version == 'xlnet_base':
            self.victim_model = XFSC(self.args)
        elif self.args.victim_model_version in ['gpt2_small', 'gpt2_medium']:
            self.victim_model = GPT2FSC(self.args)
        if torch.cuda.is_available():
            logging.info("CUDA")
            # print("CUDA")
            if torch.cuda.device_count() > 1:
                # models = nn.DataParallel(models)
                self.victim_model = torch.nn.DataParallel(self.victim_model, device_ids=self.device_ids)
            self.victim_model.to(self.device)
        # 根据模型类型决定是否加载检查点
        if self.args.victim_model_version in ['gpt2_small', 'gpt2_medium']:
            # GPT2模型使用预训练权重，不需要加载检查点
            logging.info("GPT2模型使用预训练权重，跳过检查点加载")
            self.victim_model.eval()
        else:
            # BERT、RoBERTa、XLNet模型加载检查点
            checkpoint = torch.load(
                os.path.join(self.args.saved_model_path, (self.args.task_name + self.args.victim_model_checkpoint)),
                map_location=self.device, weights_only=True)
            self.victim_model.load_state_dict(checkpoint)
            # victim_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
            self.victim_model.eval()

        if self.args.victim_model_version == 'bert_base_uncased':
            self.tokenizer = BertTokenizer.from_pretrained(self.args.victim_bert_vocab_path,
                                                           do_lower_case=self.args.do_lower_case)
        elif self.args.victim_model_version == 'roberta_base':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.args.victim_roberta_vocab_path)
        elif self.args.victim_model_version == 'xlnet_base':
            self.tokenizer = XLNetTokenizer.from_pretrained(self.args.victim_xlnet_vocab_path)
        elif self.args.victim_model_version == 'gpt2_small':
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.args.victim_model_version == 'gpt2_medium':
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if my_args is not None:
            self.args = my_args
        for key, val in kwargs.items():
            # logging.info(f'key: {key}, val: {val}')
            setattr(self.args, key, val)
        # logging.info(self.args)
        # raise SystemError

    def generate_query_and_get_predict_label(self, args=args):
        thief_data = get_pool_data(args)
        logging.info(f"original thief_data len: {len(thief_data)}")
        logging.info("generating query and geting labels with api......")
        # print("original thief_data len:", len(thief_data))
        # print("generating query and geting labels with api......")
        query = []
        predict_label = []
        sample_num = args.query_num
        data = []
        vocab, freq, probs = read_wikitext103_vocab(args)
        vocab = vocab[:10000]
        if args.initial_sample_method == 'random_sentence':
            index = random.sample(range(len(thief_data)), sample_num)
            for i in index:
                data.append(thief_data[i].strip())
        elif args.initial_sample_method == 'data_reduction_kmeans':
            data = get_reduced_data(thief_data, sample_num, args)
        elif args.initial_sample_method == 'RANDOM' or args.initial_sample_method == 'WIKI':
            result = gen_query_google_baseline(args)
            if args.task_name == 'SST-2' or \
                    args.task_name == 'IMDB' or \
                    args.task_name == 'AGNEWS' or \
                    args.task_name == 'HATESPEECH':
                data = result['sentence']

        for qi in data:
            encoded_pair = self.tokenizer(qi,
                                          padding='max_length',
                                          truncation=True,
                                          add_special_tokens=True,
                                          max_length=args.tokenize_max_length,
                                          return_tensors='pt')
            if args.victim_model_version == 'bert_base_uncased':
                token_type_ids = to(encoded_pair['token_type_ids'].squeeze(1))
            else:
                token_type_ids = None
            output = self.victim_model(input_ids=to(encoded_pair['input_ids'].squeeze(1)),
                                       token_type_ids=token_type_ids,
                                       attention_mask=to(encoded_pair['attention_mask'].squeeze(1)),
                                       train_labels=to(torch.zeros(1, args.num_labels)))
            logits = output.logits
            logits = torch.softmax(logits, 1)
            _, test_argmax = torch.max(logits, 1)
            label = test_argmax.squeeze().cpu().data.numpy()
            query.append(qi)
            predict_label.append(label)
        logging.info("sample finish.")
        # print("sample finish.")
        # if args.do_data_class_balance:
        #     query, predict_label = do_banlace(query, predict_label, args.num_labels)
        #     logging.info("oversample after")

        label_num_cnt = []
        for iii in range(args.num_labels):
            label_num_cnt.append(0)
        for i in predict_label:
            label_num_cnt[i] = label_num_cnt[i] + 1
        global label_result
        for i, j in enumerate(label_num_cnt):
            label_result = label_result + str(j)
            logging.info(f"{i} label count: {j}")
            # print("%d label count: %d" % (i, j))

        return query, predict_label

    def only_get_predict_label(self, args=args):
        query = []
        predict_label = []
        query_with_label = read_query(args)
        data = query_with_label['sentence']
        for qi in data:
            encoded_pair = self.tokenizer(qi,
                                          padding='max_length',
                                          truncation=True,
                                          add_special_tokens=True,
                                          max_length=args.tokenize_max_length,
                                          return_tensors='pt')
            if args.victim_model_version == 'bert_base_uncased':
                token_type_ids = to(encoded_pair['token_type_ids'].squeeze(1))
            else:
                token_type_ids = None
            output = self.victim_model(input_ids=to(encoded_pair['input_ids'].squeeze(1)),
                                       token_type_ids=token_type_ids,
                                       attention_mask=to(encoded_pair['attention_mask'].squeeze(1)),
                                       train_labels=to(torch.zeros(1, args.num_labels)))
            logits = output.logits
            logits = torch.softmax(logits, 1)
            _, test_argmax = torch.max(logits, 1)
            label = test_argmax.squeeze().cpu().data.numpy()
            query.append(qi)
            predict_label.append(label)
        logging.info("sample finish.")
        # print("sample finish.")
        # if args.do_data_class_balance:
        #     query, predict_label = do_banlace(query, predict_label, args.num_labels)
        #     logging.info("oversample after")

        label_num_cnt = []
        for iii in range(args.num_labels):
            label_num_cnt.append(0)
        for i in predict_label:
            label_num_cnt[i] = label_num_cnt[i] + 1
        global label_result
        for i, j in enumerate(label_num_cnt):
            label_result = label_result + str(j)
            logging.info(f"{i} label count: {j}")
            # print("%d label count: %d" % (i, j))

        return query, predict_label

    def generate_query(self):
        setup_seed(args.run_seed)
        query, predict_label = self.generate_query_and_get_predict_label(args=self.args)
        # query, predict_label = only_get_predict_label() # if the query have been sampled but the labels are not predicted
        write_query(query, predict_label, args)


if __name__ == "__main__":
    # setup_seed(args.run_seed)
    # query, predict_label = generate_query_and_get_predict_label()
    # # query, predict_label = only_get_predict_label() # if the query have been sampled but the labels are not predicted
    # write_query(query, predict_label, args)
    pass
