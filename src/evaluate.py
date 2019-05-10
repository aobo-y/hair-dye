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
import matplotlib.pyplot as plt

import config
from loss import iou_loss, HairMattingLoss, acc_loss, F1_loss
from utils import create_multi_figure

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

hairmat_loss = HairMattingLoss(config.GRAD_LOSS_LAMBDA)

def evalTest(test_data, model):
    testloader = DataLoader(test_data, batch_size=4, shuffle=False)

    total_loss, total_iou, total_acc, total_f1 = 0, 0, 0, 0
    for batch in testloader:
        image, mask = (i.to(DEVICE) for i in batch)

        pred = model(image)
        total_loss += hairmat_loss(pred, mask, image).item()
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

    if absolute:
        pred[pred > .5] = 1.
        pred[pred <= .5] = 0.
    else:
        pred[pred < .4] = 0

    rows = [[img[0], pred[0]]]
    create_multi_figure(rows, dye=True)
    plt.show()


def evaluate(test_data, model, num, absolute=True):
    rows = [None] * num
    for i in range(num):
        idx = random.randint(0, len(test_data) - 1)

    # for i, idx in enumerate([
    #     203, 159, 153, 154
    # ]):
        image, mask = (i.to(DEVICE).unsqueeze(0) for i in test_data[idx])
        pred = model(image)

        loss = hairmat_loss(pred, mask, image)
        print(idx, 'loss:', loss.item())


        if absolute:
            pred[pred > .5] = 1.
            pred[pred <= .5] = 0.
        else:
            pred[pred < .4] = 0

        rows[i] = [image[0], mask[0], pred[0]] # get batch

    create_multi_figure(rows, dye=True)
    plt.show()
