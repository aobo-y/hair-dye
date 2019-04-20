"""
Evaluate
"""

import re
import math
import datetime
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt

import config
from loss import iou_loss, hairmat_loss
from utils import create_figure


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def test(test_data, model):
    testloader = DataLoader(test_data, batch_size=16, shuffle=False)

    total_loss, total_iou = 0, 0
    for batch in testloader:
        image, mask = (i.to(DEVICE) for i in batch)

        pred = model(image)
        total_loss += hairmat_loss(pred, image, mask)
        total_iou += iou_loss(pred, mask)

    print('Testing Loss: ', total_loss / len(test_data))
    print('Testing IOU: ', total_loss / len(test_data))

def evaluate(test_data, model, idx):
    image, mask = (i.to(DEVICE).unsqueeze(0) for i in test_data[idx])
    pred = model(image)

    loss = hairmat_loss(pred, image, mask).item()
    print('loss:', loss)

    pred = pred[0].argmax(0).float()

    create_figure(image[0], mask[0], pred)
    plt.show()
