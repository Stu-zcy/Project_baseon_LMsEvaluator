import os
import pandas as pd
from datasets import load_dataset
from attack.RLMI.rlmi_utilities import get_most_frequent_templates
from transformers import GPT2LMHeadModel, GPT2Tokenizer

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rlmi_attack_path = os.path.dirname(os.path.abspath(__file__))


def evaluate():
    # 加载微调后的gpt模型
    model_path = os.path.join(rlmi_attack_path, "model", "trl_emotion_tinybert4_0")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载目标模型的微调数据集
    # private_dataset_path = os.path.join(project_path, "datasets", "emotion", "private_train.csv")
    # private_dataset =  load_dataset('csv', data_files=private_dataset_path)['train']

    target_dataset_path = os.path.join(project_path, "datasets", "emotion", "private_train.csv")
    target_dataset = pd.read_csv(target_dataset_path)
    target_label_texts = target_dataset[target_dataset['label'] == 0]['text'].tolist()

    # 获取queries
    public_dataset_path = os.path.join(project_path, "datasets", "emotion", "public_train.csv")
    public_dataset = load_dataset('csv', data_files=public_dataset_path)['train']
    frequent_unigrams, frequent_bigrams, frequent_trigrams = get_most_frequent_templates(public_dataset['text'])
    templates = frequent_trigrams

    # GPT2生成文本
    texts = []
    for template in templates:
        input_ids = tokenizer.encode(template, return_tensors='pt')
        outputs = model.generate(input_ids, max_length=512, num_return_sequences=1)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        texts.append(text)

    # 找到对应的文本
    def match_target_text(generated_text, target_label_texts):
        best_count, best_text = 0, None
        for text in target_label_texts:
            count = 0
            for word in generated_text.split():
                if word in text.split():
                    count += 1
            if count > best_count:
                best_count = count
                best_text = text
        return best_text

    def evaluate_similarity(generated_text, matched_target_text):
        rr = sum([word in generated_text.split() for word in matched_target_text.split()]) / len(generated_text.split())
        # r1, r2, r3 = metric.compute(predictions=[generated_text], references=[matched_target_text]).values()
        return rr

    results = []
    rrs = []
    for text in texts:
        target_text = match_target_text(text, target_label_texts)
        similarity = evaluate_similarity(text, target_text)
        result = {'rec text': text, 'target text': target_text, 'recover rate': similarity}
        results.append(result)
        rrs.append(similarity)

    print(results)
    print(sum(rrs) / len(rrs))


if __name__ == "__main__":
    evaluate()
