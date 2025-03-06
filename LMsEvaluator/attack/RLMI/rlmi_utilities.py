from nltk import ngrams
from collections import Counter


# 生成n-gram、统计频率
def get_ngrams(texts, n):
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        all_ngrams.extend(ngrams(tokens, n))
    return all_ngrams


# 筛选出频率大于20的短语
def filter_frequent_ngrams(ngrams, threshold):
    ngram_counts = Counter(ngrams)
    return [' '.join(ngram) for ngram, count in ngram_counts.items() if count > threshold]


# 获取最频繁的模板
def get_most_frequent_templates(texts):
    unigrams = get_ngrams(texts, 1)
    bigrams = get_ngrams(texts, 2)
    trigrams = get_ngrams(texts, 3)
    frequent_unigrams = filter_frequent_ngrams(unigrams, threshold=20)
    frequent_bigrams = filter_frequent_ngrams(bigrams, threshold=20)
    frequent_trigrams = filter_frequent_ngrams(trigrams, threshold=20)
    return frequent_unigrams, frequent_bigrams, frequent_trigrams


def calculate_ngram_penalty(text, n):
    words = text.split()
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    ngram_freq = Counter(ngrams)
    repeated_ngrams = sum(count - 1 for count in ngram_freq.values() if count > 1)
    return repeated_ngrams


if __name__ == "__main__":
    # text = 'i i i feel like'
    # print(calculate_ngram_penalty(text, 1))
    pass
