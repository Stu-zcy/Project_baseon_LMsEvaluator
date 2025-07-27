# get possible words
def get_word_set(tokenizer, t_grads_token):
    used_ids = []
    for i in range(tokenizer.vocab_size):
        if t_grads_token[i].abs().sum() != 0:
            used_ids += [i]

    used_tokens = []
    for i in used_ids:
        if (i != 101) & (i != 102):
            used_tokens.append(tokenizer.decode(i))
    return used_tokens


# get word ids
def get_id_set(tokenizer, t_grads_token):
    used_ids = []
    for i in range(tokenizer.vocab_size):
        if (i != 101) & (i != 102):
            if t_grads_token[i].abs().sum() != 0:
                used_ids += [i]
    return used_ids


# get sentence length
def get_length(t_grads_position):
    count = 0
    for i in t_grads_position:
        if i.abs().sum() != 0:
            count += 1
    return count


# distance function l2
def l2(grads1, grads2):
    l2 = 0.0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            l2 += (g1 - g2).square().sum()
    return l2


# distance function l1
def l1(grads1, grads2):
    l1 = 0.0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            l1 += (g1 - g2).abs().sum()
    return l1


# distance function cos
def cos(grads1, grads2):
    cos = 0
    n_g = 0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            cos += 1.0 - (g1 * g2).sum() / (g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2))
            n_g += 1
    cos /= n_g
    return cos


# calculate edit distance
def get_edit_distance(reference, prediction):
    len_reference = len(reference) + 1
    len_prediction = len(prediction) + 1

    dp = [[0] * len_prediction for _ in range(len_reference)]

    for i in range(len_reference):
        dp[i][0] = i
    for j in range(len_prediction):
        dp[0][j] = j

    for i in range(1, len_reference):
        for j in range(1, len_prediction):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # delete
                           dp[i][j - 1] + 1,  # insert
                           dp[i - 1][j - 1] + cost)  # replace

    distance = dp[-1][-1]
    return distance
