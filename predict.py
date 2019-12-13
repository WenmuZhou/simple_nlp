# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun

import os
import time
import torch

from model import BiRNN, TextCNN


def pad(x, max_l):
    return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))


class Pytorch_model:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.vocab = checkpoint['vocab']
        # net = BiRNN(len(self.vocab), embed_size=100, num_hiddens=100, num_layers=2)
        net = TextCNN(len(self.vocab), embed_size=100, kernel_sizes=[3, 4, 5], num_channels=[100, 100, 100])
        self.model = net
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence):
        tic = time.time()
        if isinstance(sentence, str):
            sentence = sentence.split(' ')
        sentence = torch.tensor(pad([self.vocab.stoi[word] for word in sentence], 500), device=self.device)
        sentence = sentence.unsqueeze(0)
        with torch.no_grad():
            preds = self.model(sentence).softmax(dim=1)[0]
        label = preds.argmax().item()
        conf = preds[label].item()
        result = 'positive' if label == 1 else 'negative'
        return {'value': result, 'conf': conf, 'time': time.time() - tic}


if __name__ == '__main__':
    'this movie is so great'

    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    model_path = 'output/best_model.pth'
    # 初始化网络
    model = Pytorch_model(model_path, gpu_id=0)
    while True:
        s = input('请输入字符串：')
        if s == 'exit':
            break
        result = model.predict(s)
        print(result)
