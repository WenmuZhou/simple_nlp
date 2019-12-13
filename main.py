# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 14:57
# @Author  : zhoujun
import os
import shutil
import time
import torch
from torch import nn
import torch.utils.data as Data

from dataset import get_dataset
from model import BiRNN, TextCNN


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train():
    output_foder = 'output'
    shutil.rmtree(output_foder, ignore_errors=True)
    if not os.path.exists(output_foder):
        os.makedirs(output_foder)

    device = torch.device('cuda:0')
    batch_size = 64
    train_set, test_set, vocab = get_dataset('data/aclImdb', max_l=500)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_set, batch_size)
    # net = BiRNN(len(vocab), embed_size=100, num_hiddens=100, num_layers=2)
    net = TextCNN(len(vocab), embed_size=100, kernel_sizes=[3, 4, 5], num_channels=[100, 100, 100])
    lr, num_epochs = 0.01, 5
    log_iter = 10
    # 要过滤掉不计算梯度的embedding参数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    best_metric = {'acc': 0}
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        batch_start = time.time()
        for i, (X, labels) in enumerate(train_loader):
            lr = optimizer.param_groups[0]['lr']
            cur_batch_size = X.size()[0]
            X, labels = X.to(device), labels.to(device)
            preds = net(X)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算指标
            train_l_sum += loss.cpu().item()
            cur_acc_num = (preds.argmax(dim=1) == labels).sum().cpu().item()
            train_acc_sum += cur_acc_num
            n += labels.shape[0]
            batch_count += 1
            if i % log_iter == 0:
                batch_time = time.time() - batch_start
                print('[{}/{}], [{}/{}], Speed: {:.1f} samples/sec, acc: {:.4f}, lr:{:.6}, time:{:.2f}'.format(
                    epoch + 1, num_epochs, i + 1, len(train_loader), log_iter * cur_batch_size / batch_time, cur_acc_num / cur_batch_size, lr, batch_time))
                batch_start = time.time()
        test_acc = evaluate_accuracy(test_loader, net)
        print('epoch {}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}, time {:.1f} sec'.
              format(epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if test_acc > best_metric['acc']:
            save_dict = {
                'state_dict': net.state_dict(),
                'vocab': vocab
            }
            torch.save(save_dict, '{}/best_model.pth'.format(output_foder))


if __name__ == '__main__':
    train()
