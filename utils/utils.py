import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
import re
import datetime

from PIL import Image
import torch
import cv2
import json

class AverageMeter(object):

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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_network(args):
    if args.arch == 'PCXRNet34':
        from models.PCXRNet import PCXRNet34
        net = PCXRNet34(num_class=args.num_class)   

    else:
        print('the network is not supported')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net
    
    
    
    
