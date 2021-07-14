import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.backends.cudnn as cudnn

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import numpy as np
import utils
import argparse
from utils import get_network

from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

parser = argparse.ArgumentParser(description='model test')
parser.add_argument('--test_root', required=True, help='path to val dataset (images list file)')
parser.add_argument('--model_path', type=str, default='weights', help='model file to save')
parser.add_argument('--model_name', type=str, default='name', help='model file to save')
parser.add_argument('--gpu', type=str, default='0', help='ID of GPUs to use, eg. 1,3')
parser.add_argument('--arch', '-a', metavar='ARCH', default='se_resnet101', help='model architecture')
parser.add_argument('--num_class', type=int, default=365, help='model file to resume to train')
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--image_size', type=int, default=224, help='image size')
args = parser.parse_args()
print(args)

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = list(range(len(args.gpu.split(','))))
else:
    gpus = [0] 

image_size = args.image_size
normalize = transforms.Normalize(mean=[0.4815, 0.4815, 0.4815],
                                 std=[0.2235, 0.2235, 0.2235])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.test_root, transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True)

model = get_network(args)
model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
cudnn.benchmark = True

model_path = args.model_path
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

criterion = nn.CrossEntropyLoss().cuda()

test_accuracy = []
y_ture = []
y_output = []
y_pred = []
def validate(val_loader, model, criterion):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.eval()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(gpus[0], async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1 = utils.accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        
        target = torch.flatten(target)
        target = target.cpu().numpy().tolist()
        y_ture.append(target)

        score_tmp = output
        output_value = score_tmp.detach().cpu().numpy().tolist()
        y_output.append(output_value)

        _, pred = output.data.topk(1, 1, True, True)
        pred = torch.flatten(pred)
        pred = pred.cpu().numpy().tolist()
        y_pred.append(pred)

        

        if (i+1) % args.displayInterval == 0:
            print('test: ({0}/{1})\t'
                  'Loss {loss.avg:.4f}\t'
                  'Prec_1 {top1.avg:.3f}'.format(
                      i, len(val_loader), loss=losses, top1=top1))

    test_accuracy.append(round(float(top1.avg), 3))

    return top1.avg

validate(val_loader, model, criterion)

y_ture = np.array(sum(y_ture, []))
y_pred = np.array(sum(y_pred, []))
y_output = np.array(sum(y_output, []))

fpr, tpr, thresholds = roc_curve(y_ture, y_output[:, 1])
AUC = auc(fpr, tpr)

acc_value = acc(y_ture, y_pred)
acc_value = round(acc_value, 5)

precision_mac_value = precision(y_ture, y_pred, average="macro")
precision_mic_value = precision(y_ture, y_pred, average="micro")
precision_wei_value = precision(y_ture, y_pred, average="weighted")
precision_list = []
precision_list.append(round(precision_mac_value, 5))
precision_list.append(round(precision_mic_value, 5))
precision_list.append(round(precision_wei_value, 5))

recall_mac_value = recall(y_ture, y_pred, average="macro")
recall_mic_value = recall(y_ture, y_pred, average="micro")
recall_wei_value = recall(y_ture, y_pred, average="weighted")
recall_list = []
recall_list.append(round(recall_mac_value, 5))
recall_list.append(round(recall_mic_value, 5))
recall_list.append(round(recall_wei_value, 5))

f1_mac_value = f1(y_ture, y_pred, average="macro")
f1_mic_value = f1(y_ture, y_pred, average="micro")
f1_wei_value = f1(y_ture, y_pred, average="weighted")
f1_list = []
f1_list.append(round(f1_mac_value, 5))
f1_list.append(round(f1_mic_value, 5))
f1_list.append(round(f1_wei_value, 5))

print()
print("{: ^20} {:.5f}".format("acc_value", acc_value))
print("{: ^20} {:.5f}\t{: ^20} {:.5f}\t{: ^20} {:.5f}".format("f1_macro", f1_mac_value, "f1_micro", f1_mic_value, "f1_weighted", f1_wei_value))
print("{: ^20} {:.5f}\t{: ^20} {:.5f}\t{: ^20} {:.5f}".format("recall_macro", recall_mac_value, "recall_micro", recall_mic_value, "recall_weighted", recall_wei_value))
print("{: ^20} {:.5f}\t{: ^20} {:.5f}\t{: ^20} {:.5f}".format("precision_macro", precision_mac_value, "precision_micro", precision_mic_value, "precision_weighted", precision_wei_value))
#print("{: ^20} {:.5f}".format("AUC_value", AUC))

