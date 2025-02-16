import jieba
import pandas as pd
import gensim
from gensim.models import Word2Vec

# 加载停用词
def load_stop_words(file="stopwords.txt"):
    with open(file, "r", encoding="utf-8") as f:
        return f.read().split("\n")

# 分词处理
def cut_words(file="数学原始数据.csv"):
    stop_words = load_stop_words()
    result = []
    all_data = pd.read_csv(file, encoding="gbk", names=["data"])["data"]
    for words in all_data:
        c_words = jieba.lcut(words)
        result.append([word for word in c_words if word not in stop_words])
    return result

# 训练 Gensim Word2Vec CBOW 模型
def train_word2vec(data, embedding_dim=107, window=3, min_count=1, workers=4):
    # 使用 Gensim 训练 Word2Vec 模型（CBOW）
    model = Word2Vec(sentences=data,
                     vector_size=embedding_dim,  # 嵌入维度
                     window=window,              # 上下文窗口大小
                     min_count=min_count,        # 最小词频
                     sg=0,                       # sg=0 表示使用 CBOW，sg=1 使用 Skip-Gram
                     workers=workers)            # 使用的并行工作线程数
    return model

if __name__ == "__main__":
    # 加载数据并分词
    data = cut_words()

    # 训练 Word2Vec CBOW 模型
    model = train_word2vec(data)

    # 保存模型
    model.save("word2vec_cbow_gensim.model")

    # 使用训练好的模型
    word_vectors = model.wv

    # 获取某个词的词向量
    word = "分子"
    if word in word_vectors:
        print(f"词 '{word}' 的词向量：\n", word_vectors[word])

    # 查找与某个词最相似的词
    similar_words = word_vectors.most_similar("分子", topn=5)
    print("\n与 '分子' 最相似的 5 个词：")
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity}")
