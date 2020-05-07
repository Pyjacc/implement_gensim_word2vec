'''
    手动实现gensim中的skip-gram模型
    训练词向量
'''


from collections import Counter
import numpy as np
import random
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


#语料库（用于训练词向量的数据）,在第一章train_word2vec_model.py中生成
text_path = './data/sentences.txt'
save_model_path = "./data/vocab.model"


#训练数据(用于快速调试代码)
text = "You can see the results in their published model, which was trained on 100 " \
       "billion words from a Google News dataset. The addition of phrases to the model " \
       "swelled the vocabulary size to 3 million words!"


FREQ = 0                            #低频词阈值
DELETE_HIGH_FREQ_WORDS = False      #是否删除高频词
EMBEDDING_DIM = 300                 #词向量维度
EPOCHS = 1000                       #训练的轮数
BATCH_SIZE = 5                      #每一批训练数据大小
WINDOW_SIZE = 5                     #周边词窗口大小
N_SAMPLES = 3                       #负样本大小
PRINT_EVERY = 1000                  #控制打印频率


# 从语料库中读取数据,后面用于训练词向量
def load_txt(txt_path):
    text = []
    with open(txt_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line.strip()
            text.append(line)
    f.close()
    return " ".join(text)


# 文本预处理(转换为小写,去除低频词)
def preprocess(text, FREQ, text_path=text_path):
    # text += load_txt(text_path)       #快速调试时,不用去文件中读取数据,直接传入一段文字进行调试
    text = text.lower()
    words = text.split()
    words_counts = Counter(words)
    # 去除低频词,频率小于FREQ的词将被过滤掉
    filter_words = [word for word in words if words_counts[word] > FREQ]
    return filter_words


# 构建vocab和reverse_vocab
def build_vocab(words):
    words = set(words)
    vocab2index = {word:index for index, word in enumerate(words)}
    index2vocab = {index: word for index, word in enumerate(words)}
    # 将文本转化为数值,vocab2index[word]为word对应的index
    index_words = [vocab2index[word] for word in vocab2index]
    return vocab2index, index2vocab,index_words


def get_train_and_noist_words(index_words):
    '''

    :param index_words: 为字典中词的索引构成的列表
    :return: 中心词表和负采样词表
    '''

    # 计算单词词频
    words_count = Counter(index_words)
    total_count = len(index_words)
    # 词频 = 单词出现的次数 / 总单词数
    word_freq = {word: count / total_count for word, count in words_count.items()}

    if DELETE_HIGH_FREQ_WORDS:
        t = 1e-5
        prob_drop = {word: 1 - np.sqrt(t / word_freq[word]) for word in index_words}
        train_words = [word for word in index_words if random.random() < (1 - prob_drop[word])]
    else:
        train_words = index_words


    #计算被选为负采样词的概率
    # 单词词频分布,将词频转换为np.array格式用于计算
    word_freq_array = np.array(list(word_freq.values()))
    normalize_freq = word_freq_array / word_freq_array.sum()        # word_freq_array.sum() = 1
    #从numpy数组创建一个张量，数组和张量共享相同内存
    noist_dist = torch.from_numpy(normalize_freq ** 0.75 / np.sum(normalize_freq ** 0.75))
    return train_words, noist_dist


#定义模型:实现skip-gram模型,利用负采样进行优化
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist        # 负采样词

        # 定义词向量层
        # n_vocab：有多少个词（词表大小）, n_embed：词向量的维度
        self.in_embed = nn.Embedding(n_vocab, n_embed)      #输入
        self.out_embed = nn.Embedding(n_vocab, n_embed)     #输出

        # 词向量层参数初始化(随机初始化),初始化有利于更快的收敛
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    # 输入词的前向过程
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)  #在in_embed进行训练
        return input_vectors

    # 目标词的前向过程
    def forward_output(self, output_words):
        output_vector = self.out_embed(output_words)
        return output_vector

    # 负样本词的前向过程
    def forward_noise(self, size, N_SAMPLES):
        noise_dist = self.noise_dist
        # 从词汇分布中采样负样本
        # size：表示中心词的个数,也就是BATCH_SIZE的大小
        # N_SAMPLES：选取多少个负采样词(每训练一个正样本,选取多少个负样本)
        # 对noise_dist进行size * N_SAMPLES次取样,replacement=True表示有放回的取样
        noise_words = torch.multinomial(noise_dist, size * N_SAMPLES, replacement=True)
        noise_vectors = self.out_embed(noise_words)
        # view类似于reshape操作,view操作是为了后面矩阵相乘
        noise_vectors = noise_vectors.view(size, N_SAMPLES, self.n_embed)
        return noise_vectors


#定义损失函数
class NegativeSampleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        BATCH_SIZE, embed_size = input_vectors.shape
        # 将输入词向量与目标词向量作维度转化处理
        input_vectors = input_vectors.view(BATCH_SIZE, embed_size, 1)
        output_vectors = output_vectors.view(BATCH_SIZE, 1, embed_size)

        # 目标词损失(正样本损失)
        # out_loss的维度为[BATCH_SIZE,1,1]
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()   #降维,去除维度为1的维（只有该维度为1时才能去掉）

        # 负样本损失
        # neg()：取负号
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum()     #所有负样本的值求和

        return -(out_loss + noise_loss).mean()


#模型、损失函数及优化器初始化
def initialize_model_loss_optimizer(vocab2index, noise_dist):
    skip_gram_model = SkipGramNeg(len(vocab2index), EMBEDDING_DIM, noise_dist)
    criticism = NegativeSampleLoss()      #损失函数
    # model.parameters()：优化器优化的参数（此处选model的所有参数）
    # lr：学习率
    optimizer = optim.Adam(skip_gram_model.parameters(), lr=0.003)
    return skip_gram_model, criticism, optimizer


def get_output_words(words, index, WINDOW_SIZE):
    '''
    #获取中心词周围的输出词
    :param words: 一个batch的语料
    :param index: 中心词对应的Index
    :param WINDOW_SIZE: 窗口大小,设WINDEOW_SIZE = 3,则取中心词左右两边各3个词组成训练样本（共6组训练样本）
    :return:
    '''
    # 实际操作的时候，不一定会真的取窗口那么大小，而是取一个小于等于的随机数即可
    target_window = np.random.randint(1, WINDOW_SIZE+1)
    start_word = index - WINDOW_SIZE if index >= WINDOW_SIZE else 0
    end_word = index + WINDOW_SIZE
    # 不包含中心词自身
    target_words = set(words[start_word: index] + words[index+1: end_word+1])
    return target_words


#获取训练样本:(input:output)
def get_batch(words, BATCH_SIZE, WINDOW_SIZE):
    n_batches = len(words) // BATCH_SIZE
    # 可能会遇到len(words)除以BATCH_SIZE为小数的情况
    words = words[:n_batches * BATCH_SIZE]    #保证取到words中所有的词

    for index in range(0, len(words), BATCH_SIZE):
        batch_x, batch_y = [], []
        # WINDOW_SIZE的采样目标词是在BATCH_SIZE下做的，所以最好让BATCH_SIZE要大于WINDOW_SIZE,否则WINDOW_SIZE没意义
        batch = words[index: index + BATCH_SIZE]#将words分成一个一个的batch
        for i in range(len(batch)):
            x = batch[i]
            y = get_output_words(batch, i, WINDOW_SIZE)
            # 更加清晰skip gram的原理:不是一次性的一个输入对应多个输出,而是一个输入对应一个输出（且输入不变）
            # 如“我很爱你”,取中心词为爱,则[输入：输出]为[爱:我][爱:很][爱:你]
            batch_x.extend([x] * len(y))
            batch_y.extend(y)
        yield batch_x, batch_y      #组装成x:y形式,即input:output


#训练词向量
def train_model(model, train_words, criticism, optimizer):
    steps = 0
    for e in range(EPOCHS):
        for input_words, target_words in get_batch(train_words, BATCH_SIZE, WINDOW_SIZE):
            steps += 1
            # input_words为list,利用LongTensor转换为torch对应的张量,再进行计算
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)

            # 输入、输出以及负样本向量
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            size, _ = input_vectors.shape
            noise_vectors = model.forward_noise(size, N_SAMPLES)

            #计算损失
            loss = criticism(input_vectors, output_vectors, noise_vectors)
            if steps % PRINT_EVERY == 0:
                print("LOSS :",  loss)


            # 梯度回传（基本为固定用法）
            optimizer.zero_grad()   #梯度置0
            loss.backward()
            optimizer.step()


#可视化词向量
def show_vocab(model, index2vocab):
    for index, word in index2vocab.items():
        vectors = model.state_dict()["in_embed.weight"]
        x, y = float(vectors[index][0]), float(vectors[index][1])
        plt.scatter(x, y)
        plt.annotate(word, xy=(x,y), xytext=(5,2), textcoords="offset points", ha="right", va="bottom" )

    plt.show()


# def save_model(model, save_path):


if __name__ == "__main__":
    words = preprocess(text, FREQ)
    vocab2index, index2vocab, index_words = build_vocab(words)
    train_words, noise_dist = get_train_and_noist_words(index_words)
    model, criticism, optimizer = initialize_model_loss_optimizer(vocab2index, noise_dist)
    train_model(model, train_words, criticism, optimizer)
    # save_model(model, save_model_path)
    show_vocab(model, index2vocab)

