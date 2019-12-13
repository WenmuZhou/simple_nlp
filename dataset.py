# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 15:03
# @Author  : zhoujun
import os
import random
from tqdm import tqdm
import torch
import collections
import torch.utils.data as Data
import torchtext.vocab as Vocab


def read_imdb(folder='train', data_root="data/aclImdb"):
    """
    读取数据集
    """
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name), desc='load {}'.format(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


def get_tokenized_imdb(data):
    """
    基于空格进行分词。
    data: list of [string, label]
    """

    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 创建词典。我们在这里过滤掉了出现次数少于5的词
    return Vocab.Vocab(counter, min_freq=5)


def preprocess_imdb(data, vocab, max_l=500):
    # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


def load_csv(csv_file, sep=','):
    data = []
    with open(csv_file, encoding='utf8') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split(sep)
            data.append(line)
    return data


def get_dataset(data_root, max_l):
    # train_data, test_data = read_imdb('train', data_root), read_imdb('test', data_root)
    train_data = load_csv(os.path.join(data_root, 'train.csv'))
    test_data = load_csv(os.path.join(data_root, 'test.csv'))
    vocab = get_vocab_imdb(train_data)
    print('# words in vocab:', len(vocab))
    train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab, max_l))
    test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab, max_l))
    return train_set, test_set, vocab
