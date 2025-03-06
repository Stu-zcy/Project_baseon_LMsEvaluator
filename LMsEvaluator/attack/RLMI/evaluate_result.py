"""3 评估攻击效果"""
import os
import ast
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from utils.my_prettytable import MyPrettyTable
from transformers import AutoModelForSequenceClassification, AutoTokenizer

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rlmi_attack_path = os.path.dirname(os.path.abspath(__file__))


def evaluate_result(result_rewards_path=None, reward_curve_save_path=None, evaluate_model_path=None,
                    private_train_dataset_path=None, tinybert4_emotion_attack_path=None,
                    tinybert4_emotion_final_attack_path=None, result_infer_path=None, result_final_infer_path=None):
    reward_plot_show(result_rewards_path=result_rewards_path, reward_curve_save_path=reward_curve_save_path)

    similarity_plot_show(evaluate_model_path=evaluate_model_path, private_train_dataset_path=private_train_dataset_path,
                         tinybert4_emotion_attack_path=tinybert4_emotion_attack_path,
                         tinybert4_emotion_final_attack_path=tinybert4_emotion_final_attack_path,
                         result_infer_path=result_infer_path, result_final_infer_path=result_final_infer_path)


def reward_plot_show(result_rewards_path=None, reward_curve_save_path=None):
    """3.1 对reward曲线画图"""
    if result_rewards_path is None:
        result_rewards_path = os.path.join(rlmi_attack_path, "result", "tinybert4_emotion_mean_rewards.csv")
    if reward_curve_save_path is None:
        reward_curve_save_path = os.path.join(rlmi_attack_path, "result", "reward_curve.eps")

    df = pd.read_csv(result_rewards_path, header=None)
    rewards = [ast.literal_eval(tensor.strip('tensor()')) for tensor in df.iloc[0]]

    # 绘制rewards曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Rewards曲线')
    plt.xlabel('步骤')
    plt.ylabel('Reward值')
    plt.grid(True)
    plt.savefig(reward_curve_save_path)
    plt.show()


def similarity_plot_show(evaluate_model_path=None, private_train_dataset_path=None, tinybert4_emotion_attack_path=None,
                         tinybert4_emotion_final_attack_path=None, result_infer_path=None,
                         result_final_infer_path=None):
    """3.2 评估相似度"""

    # 定义相似度标准，word recovery rate
    def evaluate_similarity(generated_text, matched_target_text):
        count = 0
        for word in matched_target_text.split():
            if word in generated_text.split():
                count += 1
        rr = count / len(matched_target_text.split())
        return rr

    # 加载评估模型（目标模型）
    # if evaluate_model_path is None:
    #     evaluate_model_path = os.path.join(rlmi_attack_path, "model", "tinybert4_emotion_attack")

    evaluate_model_path = os.path.join(rlmi_attack_path, "model", "tinybert4_emotion_attack")
    evaluate_model = AutoModelForSequenceClassification.from_pretrained(evaluate_model_path)
    tokenizer = AutoTokenizer.from_pretrained(evaluate_model_path)
    evaluate_model.eval()

    # 目标数据集中指定标签的文本
    target_label = 0
    # if private_train_dataset_path is None:
    #     private_train_dataset_path = os.path.join(project_path, "datasets", "emotion", "private_train_dataset.csv")

    private_train_dataset_path = os.path.join(project_path, "datasets", "emotion", "private_train_dataset.csv")
    private_train_dataset = load_dataset('csv', data_files=private_train_dataset_path)['train']
    target_texts = []
    for sample in private_train_dataset:
        if sample['label'] == target_label:
            target_texts.append(sample['text'])
    # print(len(target_texts)) # 19259

    """3.2.1 对训练过程中的文本进行相似度评估"""
    # 加载攻击模型训练过程中的最优状态的数据集并评估
    # if tinybert4_emotion_attack_path is None:
    #     tinybert4_emotion_attack_path = os.path.join(rlmi_attack_path, "result", "tinybert4_emotion_attack.csv")

    tinybert4_emotion_attack_path = os.path.join(rlmi_attack_path, "result", "tinybert4_emotion_attack.csv")
    df = pd.read_csv(tinybert4_emotion_attack_path, header=0)
    attack_queries = df.iloc[:, 2].tolist()
    attack_responses = df.iloc[:, 3].tolist()
    rewards = df.iloc[:, 4].tolist()

    attack_results = []
    for i in range(len(attack_queries)):
        if pd.notna(attack_responses[i]):
            text = attack_queries[i] + attack_responses[i]
            result = (text, rewards[i])
            attack_results.append(result)
        else:
            text = attack_queries[i]
            result = (text, rewards[i])
            attack_results.append(result)

    attack_results_set = set(attack_results)
    attack_results_list = list(attack_results_set)
    sorted_attack_results = sorted(attack_results_list, key=lambda x: x[1], reverse=True)

    # 存储最终结果1，取前N个重建样本
    N = 1000
    selected_attack_results = sorted_attack_results[:N]

    # if tinybert4_emotion_final_attack_path is None:
    #     tinybert4_emotion_final_attack_path = os.path.join(rlmi_attack_path, "result",
    #                                                        "tinybert4_emotion_final_attack.csv")

    tinybert4_emotion_final_attack_path = os.path.join(rlmi_attack_path, "result", "tinybert4_emotion_final_attack.csv")
    with open(tinybert4_emotion_final_attack_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['target text', 'reconstructed text', 'similarity'])

    count_correct = 0
    similarities = []
    for i in range(len(selected_attack_results)):
        generated_text = selected_attack_results[i][0]

        # 攻击准确率
        inputs = tokenizer(generated_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = evaluate_model(**inputs)
            logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1)[0].item()

        if predicted_label == target_label:
            count_correct += 1

        best_similarity = 0
        best_target_text = None
        for target_text in target_texts:
            similarity = evaluate_similarity(generated_text, target_text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_target_text = target_text
        with open(tinybert4_emotion_final_attack_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([best_target_text, generated_text, best_similarity])
        similarities.append(best_similarity)

    average_asr_during_attack = count_correct / len(selected_attack_results)
    average_wer_during_attack = sum(similarities) / len(similarities)

    print('攻击成功率为', f"{average_asr_during_attack:.2%}")
    print('平均单词恢复率为', f"{average_wer_during_attack:.5f}")

    """3.2.2 对推理结果进行评估"""

    # if result_infer_path is None:
    #     result_infer_path = os.path.join(rlmi_attack_path, "result", "tinybert4_emotion_infer.csv")
    # if result_final_infer_path is None:
    #     result_final_infer_path = os.path.join(rlmi_attack_path, "result", "tinybert4_emotion_final_infer.csv")

    result_infer_path = os.path.join(rlmi_attack_path, "result", "tinybert4_emotion_infer.csv")
    result_final_infer_path = os.path.join(rlmi_attack_path, "result", "tinybert4_emotion_final_infer.csv")

    df = pd.read_csv(result_infer_path, header=0)
    infer_texts = df.iloc[:, 0].tolist()
    rewards = df.iloc[:, 1].tolist()
    infer_results = [(infer_texts[i], rewards[i]) for i in range(len(infer_texts))]
    sorted_infer_results = sorted(infer_results, key=lambda x: x[1], reverse=True)
    # print(sorted_infer_results)

    N = 1000
    selected_infer_results = sorted_infer_results[:N]

    with open(result_final_infer_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['target text', 'reconstructed text', 'similarity'])

    similarities = []
    count_correct = 0
    for i in range(len(selected_infer_results)):
        generated_text = selected_infer_results[i][0]
        # 攻击准确率
        inputs = tokenizer(generated_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = evaluate_model(**inputs)
            logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1)[0].item()

        if predicted_label == target_label:
            count_correct += 1

        # 相似度
        best_similarity = 0
        best_target_text = None
        for target_text in target_texts:
            similarity = evaluate_similarity(generated_text, target_text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_target_text = target_text
        with open(result_final_infer_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([best_target_text, generated_text, best_similarity])
        similarities.append(best_similarity)

    average_asr_during_infer = count_correct / len(selected_infer_results)
    average_wer_during_infer = sum(similarities) / len(similarities)

    print('攻击成功率为', f"{average_asr_during_infer:.2%}")
    print('平均单词恢复率为', f"{average_wer_during_infer:.5f}")

    table = MyPrettyTable()
    table.add_field_names(['RLMI Attack Results', ''])
    table.add_row(['Aver ASR During Attack:', f"{average_asr_during_attack:.2%}"])
    table.add_row(['Aver WER During Attack:', f"{average_wer_during_attack:.5f}"])
    table.add_row(['Aver ASR During Inference:', f"{average_asr_during_infer:.2%}"])
    table.add_row(['Aver WER During Inference:', f"{average_wer_during_infer:.5f}"])
    table.set_align('RLMI Attack Results', 'l')
    table.set_align('', 'l')
    table.print_table()
    table.logging_table()


if __name__ == "__main__":
    # reward_plot_show()
    similarity_plot_show()
