"""
Evaluate
"""

import re
import math
import datetime
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

import config
from loss import iou_loss, hairmat_loss


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def test(test_data, model):
    testloader = DataLoader(test_data, batch_size=16, shuffle=False)

    total_loss, total_iou = 0, 0
    for batch in testloader:
        image, mask = (i.to(DEVICE) for i in batch)

        pred = model(image, mask)
        total_loss += hairmat_loss(pred, image, mask)
        total_iou += iou_loss(pred, mask)

    print('Testing Loss: ', total_loss / len(test_data))
    print('Testing IOU: ', total_loss / len(test_data))
