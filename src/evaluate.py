"""
Evaluate
"""

import re
import math
import datetime
import random
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt

import config
from loss import iou_loss, hairmat_loss, acc_loss, F1_loss
from utils import create_multi_figure


USE_CUDA = torch.cuda.is_available()
#DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DEVICE = torch.device('cpu')

def evalTest(test_data, model):
    testloader = DataLoader(test_data, batch_size=4, shuffle=False)

    total_loss, total_iou, total_acc, total_f1 = 0, 0, 0, 0
    for batch in testloader:
        image, mask = (i.to(DEVICE) for i in batch)

        pred = model(image)
        total_loss += hairmat_loss(pred, image, mask).item()
        iloss = iou_loss(pred, mask).item()
        total_iou += iloss
        aloss = acc_loss(pred, mask).item()
        total_acc += aloss
        floss = F1_loss(pred, mask).item()
        total_f1 += floss

    print('Testing Loss: ', total_loss / len(testloader) )
    print('Testing IOU: ', total_iou / len(testloader)  )
    print('Testing Acc: ', total_acc / len(testloader)  )
    print('Testing F1: ', total_f1 / len(testloader)  )

def evaluateOne(img, model, absolute=True):
    img = img.to(DEVICE).unsqueeze(0)
    pred = model(img)

    pred = pred.squeeze() # remove batch dim

    if absolute:
        pred = pred.argmax(0)
    else:
        pred = F.softmax(pred, dim=0)
        pred = pred[1]
        pred[pred < .4] = 0

    pred = pred.float()

    rows = [[img[0], pred]]
    create_multi_figure(rows, dye=True)
    plt.show()


def evaluate(test_data, model, num, absolute=True):
    rows = [None] * num
    for i in range(num):
        idx = random.randint(0, len(test_data) - 1)
    # for i, idx in enumerate([203, 162, 53, 116, 159]):
        image, mask = (i.to(DEVICE).unsqueeze(0) for i in test_data[idx])
        pred = model(image)

        loss = hairmat_loss(pred, image, mask).item()
        print(idx, 'loss:', loss)

        pred = pred.squeeze() # remove batch dim

        if absolute:
            pred = pred.argmax(0)
        else:
            pred = F.softmax(pred, dim=0)
            pred = pred[1]
            pred[pred < .4] = 0

        pred = pred.float()

        rows[i] = [image[0], mask[0], pred]

    create_multi_figure(rows, dye=True)
    plt.show()
