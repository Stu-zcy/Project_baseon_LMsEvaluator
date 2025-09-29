#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高熵掩码防御测试文件（简化版）
测试本地BERT模型上的高熵掩码防御功能
"""

import os
import sys
import torch
import logging
import random
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def test_high_entropy_defense_manual():
    """手动测试高熵掩码防御的核心逻辑"""
    logging.info("开始手动测试高熵掩码防御")
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 本地BERT模型路径
    model_path = "LMs/bert_base_uncased"
    
    if not os.path.exists(model_path):
        logging.error(f"模型路径不存在: {model_path}")
        return
    
    try:
        # 加载分词器
        tokenizer = BertTokenizer.from_pretrained(model_path)
        logging.info(f"成功加载分词器: {model_path}")
        
        # 加载模型
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,  # 二分类任务
            output_hidden_states=True
        )
        logging.info(f"成功加载模型: {model_path}")
        
        # 移动模型到设备
        model = model.to(device)
        
        # 创建测试数据
        test_sentences = [
            "This movie is absolutely fantastic and amazing!",
            "The acting was terrible and the plot was boring.",
            "A masterpiece of modern cinema."
        ]
        
        logging.info("=" * 50)
        logging.info("测试防御前的模型")
        logging.info("=" * 50)
        
        # 测试防御前的模型
        model.eval()
        for i, sentence in enumerate(test_sentences):
            inputs = tokenizer(
                sentence, 
                return_tensors='pt', 
                truncation=True, 
                padding=True,
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_label = torch.argmax(logits, dim=1).item()
                confidence = torch.softmax(logits, dim=1).max().item()
            
            logging.info(f"样本 {i+1}: {sentence}")
            logging.info(f"  预测标签: {predicted_label}, 置信度: {confidence:.4f}")
            logging.info(f"  原始logits: {logits.cpu().numpy()}")
            logging.info("")
        
        # 使用项目中已有的高熵掩码防御函数
        logging.info("=" * 50)
        logging.info("应用项目中的高熵掩码防御")
        logging.info("=" * 50)
        
        # 导入防御函数
        from utils.defense_utils import high_entropy_mask_defense
        
        # 配置防御参数
        defense_config = {
            'type': 'high-entropy-mask',
            'mask_pool_size': 10,
            'top_k_ratio': 0.2,
            'lr': 2e-5,
            'epochs': 1,
            'batch_size': 8,
            'mixup_k': 2,
            'num_labels': 2
        }
        
        # 应用防御
        defended_model = high_entropy_mask_defense(
            model, 
            train_data=None,  # 无训练数据，使用推理时防御
            val_data=None, 
            defense_config=defense_config
        )
        
        logging.info("项目中的高熵掩码防御已应用！")
        
        # 测试防御后的模型
        logging.info("=" * 50)
        logging.info("测试防御后的模型")
        logging.info("=" * 50)
        
        defended_model.eval()
        for i, sentence in enumerate(test_sentences):
            logging.info(f"样本 {i+1}: {sentence}")
            
            # 多次运行，观察防御效果
            for run in range(3):
                inputs = tokenizer(
                    sentence, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True,
                    max_length=128
                ).to(device)
                
                with torch.no_grad():
                    outputs = defended_model(**inputs)
                    logits = outputs.logits
                    predicted_label = torch.argmax(logits, dim=1).item()
                    confidence = torch.softmax(logits, dim=1).max().item()
                
                logging.info(f"  运行 {run+1}: 预测={predicted_label}, 置信度={confidence:.4f}")
                logging.info(f"    logits: {logits.cpu().numpy()}")
            
            logging.info("")
        
        # 测试防御一致性
        logging.info("=" * 50)
        logging.info("测试防御一致性")
        logging.info("=" * 50)
        
        sample_sentence = test_sentences[0]
        logging.info(f"测试样本: {sample_sentence}")
        
        # 收集多次运行的结果
        all_logits = []
        all_predictions = []
        all_confidences = []
        
        num_runs = 10
        for run in range(num_runs):
            inputs = tokenizer(
                sample_sentence, 
                return_tensors='pt', 
                truncation=True, 
                padding=True,
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                outputs = defended_model(**inputs)
                logits = outputs.logits
                predicted_label = torch.argmax(logits, dim=1).item()
                confidence = torch.softmax(logits, dim=1).max().item()
            
            all_logits.append(logits.cpu().numpy())
            all_predictions.append(predicted_label)
            all_confidences.append(confidence)
        
        # 分析结果
        logits_array = np.array(all_logits).squeeze()
        predictions_array = np.array(all_predictions)
        confidences_array = np.array(all_confidences)
        
        logging.info(f"运行次数: {num_runs}")
        logging.info(f"预测标签变化: {predictions_array}")
        logging.info(f"置信度范围: [{confidences_array.min():.4f}, {confidences_array.max():.4f}]")
        logging.info(f"置信度标准差: {confidences_array.std():.4f}")
        
        # 计算logits的变化
        logits_std = logits_array.std(axis=0)
        logging.info(f"Logits标准差范围: [{logits_std.min():.4f}, {logits_std.max():.4f}]")
        logging.info(f"平均Logits标准差: {logits_std.mean():.4f}")
        
        logging.info("=" * 50)
        logging.info("测试完成！")
        logging.info("=" * 50)
        
    except Exception as e:
        logging.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_high_entropy_defense_manual()
