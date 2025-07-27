import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice
)


def load_model_and_tokenizer(model_name, task_type, local_model=False, project_path=""):
    """
    根据任务类型加载模型和分词器

    参数:
        model_name: 模型名称或路径
        task_type: 任务类型 (single_sentence, sentence_pair, nli, qa, ner, lm, clm, multiple_choice)
        local_model: 是否从本地加载
        project_path: 项目路径

    返回:
        tokenizer, model
    """
    # 确定模型路径
    if local_model:
        model_path = os.path.join(project_path, 'LMs', model_name)
    else:
        model_path = model_name

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 根据任务类型加载模型
    if task_type in ["single_sentence", "sentence_pair", "nli"]:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    elif task_type == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    elif task_type == "ner":
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    elif task_type == "lm":  # 掩码语言模型
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    elif task_type == "clm":  # 因果语言模型
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif task_type == "multiple_choice":
        model = AutoModelForMultipleChoice.from_pretrained(model_path)
    else:
        # 默认使用序列分类模型
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return tokenizer, model
