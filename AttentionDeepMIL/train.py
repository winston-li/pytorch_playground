from __future__ import print_function
import os
import sys
import time
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MnistBags
from model import Attention
from ckpt import save_ckpt, load_ckpt


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)        # backward pass
    print('Load Train and Test Set')
    train_loader = DataLoader(MnistBags(target_number=args.target_number,
                                        min_target_count=args.min_target_count,
                                        mean_bag_length=args.mean_bag_length,
                                        var_bag_length=args.var_bag_length,
                                        scale=args.scale,
                                        num_bag=args.num_bags_train,
                                        seed=args.seed,
                                        train=True),
                              batch_size=args.batchsize,
                              shuffle=True,
                              num_workers=n_worker,
                              pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(MnistBags(target_number=args.target_number,
                                       min_target_count=args.min_target_count,
                                       mean_bag_length=args.mean_bag_length,
                                       var_bag_length=args.var_bag_length,
                                       scale=args.scale,
                                       num_bag=args.num_bags_test,
                                       seed=args.seed,
                                       train=False),
                             batch_size=args.batchsize,
                             shuffle=False,
                             num_workers=n_worker,
                             pin_memory=torch.cuda.is_available())

    # resume checkpoint
    checkpoint = load_ckpt()
    if checkpoint:
        print('Resume training ...')
        start_epoch = checkpoint.epoch
        model = checkpoint.model
    else:
        print('Grand new training ...')
        start_epoch = 0
        model = Attention2()

    # put model to multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.999),
                           weight_decay=args.reg)

    if checkpoint:
        try:
            optimizer.load_state_dict(checkpoint.optimizer)
        except:
            print(
                '[WARNING] optimizer not restored from last checkpoint, continue without previous state'
            )
    # free checkpoint reference
    del checkpoint

    log_dir = os.path.join('logs', args.logname)
    n_cv_epoch = 1  #2
    with SummaryWriter(log_dir) as writer:
        print('\nTraining started ...')
        for epoch in range(start_epoch + 1,
                           n_epoch + start_epoch + 1):  # 1 base
            train(model, optimizer, train_loader, epoch, writer)
            if epoch % n_cv_epoch == 0:
                with torch.no_grad():
                    test(model, optimizer, test_loader, epoch, writer)
            save_ckpt(model, optimizer, epoch)
        print('\nTraining finished ...')


def train(model, optimizer, loader, epoch, writer):
    epoch_loss = AverageMeter()
    epoch_accuracy = AverageMeter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()

    batch_total_time = 0
    prev_time = time.time()
    print('')
    total = len(loader)
    for batch_i, (data, label) in enumerate(loader, 1):
        bag_label = label.long()
        data, bag_label = data.to(device), bag_label.to(device)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        Y_logits = model(data)
        # loss = F.binary_cross_entropy_with_logits(Y_logits, bag_label)
        loss = F.cross_entropy(Y_logits, bag_label)
        # backward pass
        loss.backward()
        optimizer.step()

        acc = 100 * (
            Y_logits.detach().argmax(-1) == bag_label).cpu().numpy().mean()
        epoch_loss.update(loss.item(), data.size(0))
        epoch_accuracy.update(acc, data.size(0))

        # Determine approximate time left
        batches_left = total - batch_i
        batch_total_time += (time.time() - prev_time)
        batch_mean_time = datetime.timedelta(seconds=(batch_total_time /
                                                      batch_i))
        time_left = datetime.timedelta(seconds=batches_left *
                                       (time.time() - prev_time))
        prev_time = time.time()
        # print console log
        sys.stdout.write(
            "\r[Epoch %d (train)] [Batch %d/%d] [Loss: %.4f , Acc: %.2f%%] Batch_Time: %s ETA: %s "
            % (epoch, batch_i, total, epoch_loss.avg, epoch_accuracy.avg,
               batch_mean_time, time_left))

    writer.add_scalar(f'train/accuracy', epoch_accuracy.avg, epoch)
    writer.add_scalar(f'train/loss', epoch_loss.avg, epoch)
    return


def test(model, optimizer, loader, epoch, writer):
    epoch_loss = AverageMeter()
    epoch_accuracy = AverageMeter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    batch_total_time = 0
    prev_time = time.time()
    print('')
    total = len(loader)
    for batch_i, (data, label) in enumerate(loader, 1):
        bag_label = label.long()
        data, bag_label = data.to(device), bag_label.to(device)
        Y_logits = model(data)
        # loss = F.binary_cross_entropy_with_logits(Y_logits, bag_label)
        loss = F.cross_entropy(Y_logits, bag_label)

        acc = 100 * (
            Y_logits.detach().argmax(-1) == bag_label).cpu().numpy().mean()
        epoch_loss.update(loss.item(), data.size(0))
        epoch_accuracy.update(acc, data.size(0))

        # Determine approximate time left
        batches_left = total - batch_i
        batch_total_time += (time.time() - prev_time)
        batch_mean_time = datetime.timedelta(seconds=(batch_total_time /
                                                      batch_i))
        time_left = datetime.timedelta(seconds=batches_left *
                                       (time.time() - prev_time))
        prev_time = time.time()
        # print console log
        sys.stdout.write(
            "\r[Epoch %d (test)] [Batch %d/%d] [Loss: %.4f , Acc: %.2f%%] Batch_Time: %s ETA: %s "
            % (epoch, batch_i, total, epoch_loss.avg, epoch_accuracy.avg,
               batch_mean_time, time_left))

    writer.add_scalar(f'test/accuracy', epoch_accuracy.avg, epoch)
    writer.add_scalar(f'test/loss', epoch_loss.avg, epoch)
    return


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,
                        metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--reg',
                        type=float,
                        default=1e-4,
                        metavar='R',
                        help='weight decay (default: 1e-4) ')
    parser.add_argument('--target_number',
                        type=int,
                        default=9,
                        metavar='T',
                        help='bag to be positive if containing this one more than min_target_count (default: 9)')
    parser.add_argument('--min_target_count',
                        type=int,
                        default=1,
                        metavar='MTC',
                        help='minimum count of target number, (default: 1)')
    parser.add_argument('--mean_bag_length',
                        type=int,
                        default=15,
                        metavar='ML',
                        help='average bag length (default: 15)')
    parser.add_argument('--var_bag_length',
                        type=int,
                        default=5,
                        metavar='VL',
                        help='variance of bag length (default: 5)')
    parser.add_argument('--scale',
                        type=int,
                        default=10,
                        metavar='SZ',
                        help='multiplied scale of generated bag image (default: 10)')                        
    parser.add_argument('--num_bags_train',
                        type=int,
                        default=6000,
                        metavar='NTrain',
                        help='number of bags in training set (default: 6000)')
    parser.add_argument('--num_bags_test',
                        type=int,
                        default=1000,
                        metavar='NTest',
                        help='number of bags in test set (default: 1000)')
    parser.add_argument('--batchsize',
                        type=int,
                        default=32,
                        metavar='BS',
                        help='batchsize of training (default: 32)')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--logname',
                        type=str,
                        default='mnist',
                        metavar='LOG',
                        help='tensorboard logname')

    args = parser.parse_args()

    run(args)
