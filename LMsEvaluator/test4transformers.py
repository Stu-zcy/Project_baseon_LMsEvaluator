import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import pipeline, pipelines

if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("LMs/bert_base_uncased_english")
    model = AutoModelForMaskedLM.from_pretrained("LMs/bert_base_uncased_english")
    # print(f"model: \n {model}")
    # examples = "Who was Jim Henson? Jim Henson was a [MASK]."
    # indexed_tokens = tokenizer.encode(examples)
    # print(f"indexed_tokens: \n {indexed_tokens}")
    # tokens_tensor = torch.tensor([indexed_tokens])
    # print(f"tokens_tensor: \n {tokens_tensor}")
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(tokens_tensor)
    #     print(f"outputs: \n {outputs}")
    #     print(f"outputs[0]: \n {outputs[0]}")
    #     predicted_index = torch.argmax(outputs[0][0, -1, :]).item()
    #     print(f"predicted_index: \n {predicted_index}")
    # temp = tokenizer.decode(predicted_index)
    # print(temp)
    # # predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    # # print(f"predicted_text: \n {predicted_text}")
    pipe = pipeline("fill-mask", model="LMs/bert_base_chinese")
    print(pipe("生活的真谛是[MASK]。"))
    for K, v in pipelines.SUPPORTED_TASKS.items():
        print(K, v)

