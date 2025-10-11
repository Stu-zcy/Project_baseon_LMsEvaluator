import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaModel
from transformers import XLNetTokenizer, XLNetConfig, XLNetForSequenceClassification, XLNetModel
from transformers import GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification


class BFSC(nn.Module):
    def __init__(self, args):
        super(BFSC, self).__init__()
        bert_hidden_size = args.bert_hidden_size
        config = BertConfig.from_pretrained(
            args.victim_bert_path,
            num_labels=args.num_labels,
            hidden_dropout_prob=args.bert_hidden_dropout_prob,
            output_hidden_states=args.bert_output_hidden_states
        )
        self.bert = BertForSequenceClassification.from_pretrained(args.victim_bert_path, config=config)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels):
        output = self.bert(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           labels=train_labels)
        return output


class RFSC(nn.Module):
    def __init__(self, args):
        super(RFSC, self).__init__()
        # config = RobertaConfig.from_pretrained(
        #     args.victim_roberta_path,
        #     num_labels=args.num_labels,
        #     hidden_dropout_prob=args.roberta_hidden_dropout_prob,
        #     output_hidden_states=args.roberta_output_hidden_states
        # )
        # config = RobertaConfig.from_pretrained(
        #     args.victim_roberta_path,
        #     num_labels=args.num_labels)
        # self.roberta = RobertaForSequenceClassification.from_pretrained(args.victim_roberta_path, config=config)
        self.roberta = RobertaForSequenceClassification.from_pretrained(args.victim_roberta_path, num_labels=args.num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels):
        output = self.roberta(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=train_labels)
        return output


class XFSC(nn.Module):
    def __init__(self, args):
        super(XFSC, self).__init__()
        xlnet_hidden_size = args.xlnet_hidden_size
        config = XLNetConfig.from_pretrained(
            args.victim_xlnet_path,
            num_labels=args.num_labels,
            hidden_dropout_prob=args.xlnet_hidden_dropout_prob,
            output_hidden_states=args.xlnet_output_hidden_states
        )
        self.xlnet = XLNetForSequenceClassification.from_pretrained(args.victim_xlnet_path, config=config)

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels):
        output = self.xlnet(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           labels=train_labels)
        return output


class GPT2FSC(nn.Module):
    def __init__(self, args):
        super(GPT2FSC, self).__init__()
        # 根据模型版本选择GPT2模型
        if args.victim_model_version == 'gpt2_small':
            model_name = 'gpt2'
        elif args.victim_model_version == 'gpt2_medium':
            model_name = 'gpt2-medium'
        else:
            model_name = 'gpt2'  # 默认使用gpt2
            
        # 加载配置
        config = GPT2Config.from_pretrained(model_name, num_labels=args.num_labels)
        # 确保max_position_embeddings足够大
        if config.max_position_embeddings < 1024:
            config.max_position_embeddings = 1024
        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(model_name, config=config)
        # 设置pad_token
        self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id
        print(f"GPT2FSC config max_position_embeddings: {self.gpt2.config.max_position_embeddings}")

    def forward(self, input_ids, token_type_ids, attention_mask, train_labels):
        # GPT2不需要token_type_ids，所以忽略它
        # 确保序列长度不超过模型的最大位置嵌入
        max_length = self.gpt2.config.max_position_embeddings
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        
        
        try:
            outputs = self.gpt2(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=train_labels)
        except Exception as e:
            print(f"GPT2FSC error: {e}")
            print(f"Input details: input_ids={input_ids}, attention_mask={attention_mask}")
            raise e
        
        # GPT2ForSequenceClassification返回元组，需要转换为对象格式
        if isinstance(outputs, tuple):
            # 创建一个简单的对象来包装输出
            class OutputWrapper:
                def __init__(self, logits, loss=None):
                    self.logits = logits
                    self.loss = loss
                    # 添加其他属性以兼容防御函数
                    self.hidden_states = None
                    self.attentions = None
            
            # 如果提供了labels，outputs[0]是loss，outputs[1]是logits
            if train_labels is not None:
                return OutputWrapper(outputs[1], outputs[0])
            else:
                return OutputWrapper(outputs[0])
        else:
            return outputs