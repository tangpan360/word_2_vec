import numpy as np
import pandas as pd
import pickle
import jieba
import os
from tqdm import tqdm

# 加载停用词
def load_stop_words(file = "stopwords.txt"):
    with open(file,"r",encoding = "utf-8") as f:
        return f.read().split("\n")

# 分词处理
def cut_words(file="数学原始数据.csv"):
    stop_words = load_stop_words()
    result = []
    all_data = pd.read_csv(file,encoding = "gbk",names=["data"])["data"]
    for words in all_data:
        c_words = jieba.lcut(words)
        result.append([word for word in c_words if word not in stop_words])
    return result

# 生成词汇表和 One-Hot 编码
def get_dict(data):
    index_2_word = []
    for words in data:
        for word in words:
            if word not in index_2_word:
                index_2_word.append(word)

    word_2_index = {word:index for index,word in enumerate(index_2_word)}
    word_size = len(word_2_index)

    word_2_onehot = {}
    for word,index in word_2_index.items():
        one_hot = np.zeros((1,word_size))
        one_hot[0,index] = 1
        word_2_onehot[word] = one_hot

    return word_2_index, index_2_word, word_2_onehot

# Softmax 函数
def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

if __name__ == "__main__":
    # 加载数据和生成词典
    data = cut_words()
    word_2_index, index_2_word, word_2_onehot = get_dict(data)

    word_size = len(word_2_index)
    embedding_num = 107
    lr = 0.01
    epoch = 10
    n_gram = 3  # 上下文窗口大小

    # 初始化权重
    w1 = np.random.normal(-1, 1, size=(word_size, embedding_num))
    w2 = np.random.normal(-1, 1, size=(embedding_num, word_size))

    for e in range(epoch):
        for words in tqdm(data):
            for n_index, target_word in enumerate(words):
                target_word_onehot = word_2_onehot[target_word]
                context_words = words[max(n_index - n_gram, 0):n_index] + words[n_index + 1: n_index + 1 + n_gram]

                # 计算上下文词的平均嵌入表示
                context_vectors = np.zeros((1, embedding_num))
                for context_word in context_words:
                    context_vectors += word_2_onehot[context_word] @ w1
                context_vectors /= len(context_words)

                # 计算预测值 p
                p = context_vectors @ w2
                pre = softmax(p)

                # 计算梯度并更新权重
                G2 = pre - target_word_onehot
                delta_w2 = context_vectors.T @ G2
                G1 = G2 @ w2.T
                delta_w1 = word_2_onehot[target_word].T @ G1

                w1 -= lr * delta_w1
                w2 -= lr * delta_w2

    # 保存模型
    with open("word2vec_cbow.pkl", "wb") as f:
        pickle.dump([w1, word_2_index, index_2_word], f)
