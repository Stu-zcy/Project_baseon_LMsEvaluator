import torch, random, torch.nn.functional as F, numpy as np
import logging


# CE for onehot
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


# TL
# def TL(flag_TL, model, n):
#     if flag_TL:
#         for name, param in model.named_parameters():
#             if any(f'layer.{i}' in name for i in range(n)): # freeze first n layers
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
#     else:
#         for name, param in model.named_parameters():
#             param.requires_grad = True

# TL
def TL(flag_TL, model, l):
    if flag_TL:
        for name, param in model.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
            elif 'layer.' in name:
                layer_num = int(name.split('layer.')[1].split('.')[0])
                if layer_num < l:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True


# def TL(flag_TL, model):
#     if flag_TL:
#         for name, param in model.named_parameters():
#             if 'layer.8' in name or 'layer.9' in name or 'layer.10' in name or 'layer.11' in name or 'pooler' in name or 'classifier' in name: # 最后4层和classifier
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#     else:
#         for name, param in model.named_parameters():
#             param.requires_grad = True

# TH
def get_batch(k, dataset, tokenizer, num_labels):
    indices = random.sample(range(len(dataset)), k - 1)
    samples = [dataset[i] for i in indices]
    if 'sentence1' in samples[0] and 'sentence2' in samples[0]:
        sentence1_batch = [sample['sentence1'] for sample in samples]
        sentence2_batch = [sample['sentence2'] for sample in samples]
        batch = tokenizer(
            sentence1_batch,
            sentence2_batch,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
    else:
        batch = tokenizer(
            [sample['sentence'] for sample in samples],
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
    labels = torch.tensor([sample['label'] for sample in samples])
    if num_labels >= 2:
        onehot_labels = F.one_hot(labels, num_classes=num_labels).float()
    else:
        onehot_labels = labels
    return batch, onehot_labels


def mixup(device, k, model, pooler_output, onehot_label, p_batch, p_onehot_labels):
    w = torch.randn(k, device=device)
    w = torch.softmax(w, dim=0).detach().clone()
    with torch.no_grad():
        p_outputs = model.bert(**p_batch.to(device))
        p_pooler_outputs = p_outputs.pooler_output
    p_onehot_labels = p_onehot_labels.to(device)
    new_pooler_output = w[0] * pooler_output + sum(w[i + 1] * p_pooler_outputs[i] for i in range(k - 1))
    new_onehot_label = w[0] * onehot_label + sum(w[i + 1] * p_onehot_labels[i] for i in range(k - 1))
    return new_pooler_output, new_onehot_label


def get_pool(m, hidden_size):
    pool = []
    for _ in range(m):
        sigma = torch.sign(torch.rand(hidden_size) - 0.5)
        pool.append(sigma)
    return pool


# TH2
def compute_entropy(hidden_states, num_bins=30):
    """
    计算隐藏层每个维度的信息熵
    :param hidden_states: shape (num_samples, hidden_dim) 的隐藏层输出
    :param num_bins: 直方图分桶数
    :return: 每个维度的信息熵，shape (hidden_dim,)
    """
    hidden_states = hidden_states.cpu().detach().numpy()  # 转换为 NumPy 进行统计
    hidden_dim = hidden_states.shape[1]
    entropy_values = np.zeros(hidden_dim)

    for i in range(hidden_dim):
        hist, bin_edges = np.histogram(hidden_states[:, i], bins=num_bins, density=True)
        hist = hist + 1e-10  # 避免 log(0)
        hist = hist / np.sum(hist)  # 归一化
        entropy_values[i] = -np.sum(hist * np.log2(hist))  # 计算熵

    return torch.tensor(entropy_values, dtype=torch.float32)


def select_high_entropy_indices(entropy_values, top_k_ratio=0.3):
    """
    选择信息熵最高的维度索引
    :param entropy_values: shape (hidden_dim,)
    :param top_k_ratio: 选取信息熵最高的比例（如 30%）
    :return: 高熵维度索引的列表
    """
    hidden_dim = entropy_values.shape[0]
    top_k = int(hidden_dim * top_k_ratio)
    top_indices = torch.argsort(entropy_values, descending=True)[:top_k]  # 取 Top-K 维度
    return top_indices


# high_entropy_indices = select_high_entropy_indices(entropy_values, top_k_ratio=0.5)  # 选择前30%的高熵维度

def generate_sign_mask_pool(pool_size, hidden_dim, high_entropy_indices):
    """
    生成掩码池，每个掩码仅在高熵维度上为 ±1，其余维度为 1
    :param pool_size: 掩码池大小
    :param hidden_dim: 隐藏层维度
    :param high_entropy_indices: 高熵维度索引
    :return: 掩码池 (pool_size, hidden_dim)
    """
    mask_pool = torch.ones((pool_size, hidden_dim))  # 先初始化为全 1
    random_signs = (torch.randint(0, 2, (pool_size, len(high_entropy_indices)), dtype=torch.float32) * 2 - 1)
    mask_pool[:, high_entropy_indices] = random_signs
    return mask_pool


# ================== 统一防御入口 ==================
def apply_defense(model, train_data, val_data, defense_config):
    """
    统一防御入口，根据 defense_config['type'] 自动调用对应防御方法。
    参数:
        model: 需要防御的模型对象
        train_data: 训练集数据
        val_data: 验证集数据
        defense_config: dict，包含防御类型和参数
    返回:
        防御后的模型对象
    """
    defense_type = defense_config.get('type', '').lower()

    if defense_type == 'dp-noise':
        # 差分隐私防御
        return dp_defense(model, train_data, val_data, defense_config)
    elif defense_type == 'label-smoothing':
        return label_smoothing_defense(model, train_data, val_data, defense_config)
    elif defense_type == 'early-stopping':
        return early_stopping_defense(model, train_data, val_data, defense_config)
    elif defense_type == 'mixup':
        return mixup_defense(model, train_data, val_data, defense_config)
    elif defense_type == 'high-entropy-mask':
        return high_entropy_mask_defense(model, train_data, val_data, defense_config)
    elif defense_type == 'pruning':
        return pruning_defense(model, train_data, val_data, defense_config)
    elif defense_type == 'output-perturb':
        return output_perturb_defense(model, train_data, val_data, defense_config)
    else:
        raise ValueError(f"Unknown defense type: {defense_type}")


def dp_defense(model, train_data, val_data, defense_config):
    """
    差分隐私防御实现，基于Opacus，只训练分类头
    梯度扰动
    """
    from opacus import PrivacyEngine
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer
    from tqdm import tqdm
    import random

    device = next(model.parameters()).device
    tokenizer = defense_config.get('tokenizer', None)
    if tokenizer is None:
        tokenizer_path = defense_config.get('tokenizer_path', 'bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    epsilon = defense_config.get('epsilon', 1.0)
    clip_norm = defense_config.get('clip_norm', 1.0)
    epochs = defense_config.get('epochs', 1)
    batch_size = defense_config.get('batch_size', 8)
    lr = defense_config.get('lr', 2e-5)
    delta = defense_config.get('delta', 1e-5)

    # 冻结BERT主体参数，只训练分类头
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False

    # 构造DataLoader
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def collate_fn(batch):
        texts = [s['sentence'] for s in batch]
        labels = torch.tensor([s['label'] for s in batch], dtype=torch.long)
        return texts, labels

    train_dataset = SimpleDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn)

    model.train()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=clip_norm,
    )

    for epoch in range(epochs):
        logging.info(f"\n[DP-Defense] ========== 开始第 {epoch + 1}/{epochs} 轮训练 ==========")
        with tqdm(total=len(train_loader), desc=f"DP Epoch {epoch + 1}") as pbar:
            for texts, labels in train_loader:
                inputs = tokenizer(list(texts), return_tensors='pt', truncation=True, padding=True).to(device)
                labels = labels.to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                # 关键：reduction='none'，得到每个样本的loss
                loss = loss_fn(logits, labels)
                if loss.dim() > 0:
                    loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
        logging.info(f"[DP-Defense] ========== 第 {epoch + 1} 轮训练结束 ==========")

    epsilon_spent = privacy_engine.get_epsilon(delta)
    logging.info(f"[DP-Defense] Training finished. (ε = {epsilon_spent:.2f}, δ = {delta})")
    return model


def label_smoothing_defense(model, train_data, val_data, defense_config):
    """
    标签平滑防御实现（伪代码）
    """
    smoothing = defense_config.get('label_smoothing', 0.1)
    # 训练时loss用label smoothing
    # loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)
    # 训练流程略
    # ...
    logging.info(f"[Label Smoothing Defense] smoothing={smoothing}")
    return model


def early_stopping_defense(model, train_data, val_data, defense_config):
    """
    Early Stopping防御实现（伪代码）
    """
    patience = defense_config.get('patience', 5)
    # 训练时监控val loss，patience轮无提升则停止
    # 训练流程略
    # ...
    logging.info(f"[Early Stopping Defense] patience={patience}")
    return model


def mixup_defense(model, train_data, val_data, defense_config):
    """
    Mixup防御实现（伪代码）
    """
    alpha = defense_config.get('alpha', 0.2)
    # 训练时对输入做mixup
    # 训练流程略
    # ...
    logging.info(f"[Mixup Defense] alpha={alpha}")
    return model


def high_entropy_mask_defense(model, train_data, val_data, defense_config):
    """
    基于高熵掩码+mixup的防御实现：
    1. 统计高熵维度
    2. 生成掩码池
    3. 训练时对高熵维度加mask扰动，并与其他样本做mixup
    """
    from transformers import BertTokenizer
    import torch
    import torch.nn as nn
    import random
    from tqdm import tqdm
    import torch.nn.functional as F

    device = next(model.parameters()).device
    tokenizer = defense_config.get('tokenizer', None)
    if tokenizer is None:
        tokenizer_path = defense_config.get('tokenizer_path', 'bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # 1. 统计高熵维度
    model.eval()
    all_hidden = []
    for sample in train_data[:100]:  # 取前100条做统计
        inputs = tokenizer(sample['sentence'], return_tensors='pt', truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model.bert(**inputs)
            pooler_output = outputs.pooler_output.squeeze(0)
            all_hidden.append(pooler_output.cpu())
    hidden_states = torch.stack(all_hidden)  # (num_samples, hidden_dim)
    entropy_values = compute_entropy(hidden_states)
    top_k_ratio = defense_config.get('top_k_ratio', 0.3)
    high_entropy_indices = select_high_entropy_indices(entropy_values, top_k_ratio=top_k_ratio)
    pool_size = defense_config.get('mask_pool_size', 10)
    mask_pool = generate_sign_mask_pool(pool_size=pool_size, hidden_dim=hidden_states.shape[1],
                                        high_entropy_indices=high_entropy_indices)
    logging.info(
        f"[High Entropy Mask+Mixup Defense] 掩码池shape: {mask_pool.shape}, 高熵维度数: {len(high_entropy_indices)}")

    # 2. 训练逻辑
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=defense_config.get('lr', 2e-5))
    loss_fn = nn.CrossEntropyLoss()
    epochs = defense_config.get('epochs', 1)
    batch_size = defense_config.get('batch_size', 8)
    mixup_k = defense_config.get('mixup_k', 2)  # mixup混合的样本数，含当前样本
    num_labels = defense_config.get('num_labels', 2)

    for epoch in range(epochs):
        random.shuffle(train_data)
        num_batches = (len(train_data) + batch_size - 1) // batch_size
        with tqdm(total=num_batches, desc=f"HE-Mixup Epoch {epoch + 1}") as pbar:
            for i in range(0, len(train_data), batch_size):
                batch_samples = train_data[i:i + batch_size]
                texts = [s['sentence'] for s in batch_samples]
                labels = torch.tensor([s['label'] for s in batch_samples], dtype=torch.long, device=device)
                onehot_labels = F.one_hot(labels, num_classes=num_labels).float()
                inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True).to(device)
                # 得到BERT输出
                outputs = model.bert(**inputs)
                pooler_output = outputs.pooler_output  # (batch, hidden_dim)
                # 随机选一个mask
                mask = mask_pool[random.randint(0, pool_size - 1)].to(device)
                pooler_output_masked = pooler_output.clone()
                pooler_output_masked[:, high_entropy_indices] *= mask[high_entropy_indices]
                # mixup: 对每个样本，采样k-1个样本做mixup
                mixed_outputs = []
                mixed_labels = []
                for idx in range(pooler_output_masked.size(0)):
                    # 当前样本
                    cur_output = pooler_output_masked[idx]
                    cur_label = onehot_labels[idx]
                    # 采样k-1个不同的样本
                    pool_indices = list(range(pooler_output_masked.size(0)))
                    pool_indices.remove(idx)
                    if len(pool_indices) >= mixup_k - 1:
                        sampled_indices = random.sample(pool_indices, mixup_k - 1)
                    else:
                        sampled_indices = random.choices(pool_indices, k=mixup_k - 1)
                    p_outputs = pooler_output_masked[sampled_indices]
                    p_labels = onehot_labels[sampled_indices]
                    # 用mixup函数融合
                    mixed_output, mixed_label = mixup(device, mixup_k, model, cur_output, cur_label, p_outputs,
                                                      p_labels)
                    mixed_outputs.append(mixed_output)
                    mixed_labels.append(mixed_label)
                mixed_outputs = torch.stack(mixed_outputs)  # (batch, hidden_dim)
                mixed_labels = torch.stack(mixed_labels)  # (batch, num_labels)
                # 送入分类头
                logits = model.classifier(mixed_outputs)
                loss = cross_entropy_for_onehot(logits, mixed_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
        logging.info(f"[High Entropy Mask+Mixup Defense] epoch {epoch + 1} finished.")

    return model


def pruning_defense(model, train_data, val_data, defense_config):
    """
    模型剪枝防御实现：对模型所有nn.Linear层按比例剪枝。
    支持参数：prune_ratio（0~1，剪枝比例）
    """
    import torch
    import torch.nn.utils.prune as prune
    import torch.nn as nn

    prune_ratio = defense_config.get('prune_ratio', 0.2)  # 默认剪20%
    logging.info(f"[Pruning Defense] 对所有nn.Linear层进行{prune_ratio * 100:.1f}%剪枝")

    pruned = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
            logging.info(f"[Pruning Defense] 已对{name}.weight剪枝")
            if hasattr(module, 'bias') and module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=prune_ratio)
                logging.info(f"[Pruning Defense] 已对{name}.bias剪枝")
            # 可选：移除剪枝re-param，使其变为普通参数
            prune.remove(module, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                prune.remove(module, 'bias')
            pruned = True
    if not pruned:
        logging.info("[Pruning Defense] 未找到可剪枝的nn.Linear层，未执行剪枝")
    return model


def output_perturb_defense(model, train_data, val_data, defense_config):
    """
    输出扰动防御：在推理阶段对logits加高斯噪声，防模型窃取攻击。
    参数：noise_std（float，噪声标准差，默认0.1）
    """
    import torch
    import types

    noise_std = defense_config.get('noise_std', 0.1)
    logging.info(f"[Output Perturb Defense] 推理时对logits加高斯噪声，std={noise_std}")

    # 保存原始forward
    original_forward = model.forward

    def noisy_forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            if not self.training:
                noise = torch.randn_like(logits) * noise_std
                logits = logits + noise
            # 构造新的output对象，替换logits
            from transformers.modeling_outputs import SequenceClassifierOutput
            if isinstance(outputs, SequenceClassifierOutput):
                return SequenceClassifierOutput(
                    loss=outputs.loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions
                )
            else:
                # 兼容其他输出类型
                outputs = list(outputs)
                outputs[1] = logits
                return tuple(outputs)
        else:
            return outputs

    # 绑定新forward
    model.forward = types.MethodType(noisy_forward, model)
    return model
