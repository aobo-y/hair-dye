"""
Train seq2seq
"""

import os
import math
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import config
from utils import create_figure

from loss import iou_loss, HairMattingLoss

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

class Trainer:
    '''Trainer to train the seq2seq model'''

    def __init__(self, model, checkpoint_mng):
        self.model = model

        self.checkpoint_mng = checkpoint_mng

        self.optimizer = optim.Adam(model.parameters(), lr=config.LR, eps=1e-7)

        # trained epochs
        self.trained_epoch = 0
        self.train_loss = {'loss': [], 'iou':[]}
        self.dev_loss = {'loss': [], 'iou':[]}

        self.loss = HairMattingLoss(config.GRAD_LOSS_LAMBDA).to(DEVICE)

    def log(self, *args):
        '''formatted log output for training'''

        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{time}   ', *args)

    def resume(self, checkpoint):
        '''load checkpoint'''

        self.trained_epoch = checkpoint['epoch']
        self.train_loss = checkpoint['train_loss']
        self.dev_loss = checkpoint['dev_loss']
        self.optimizer.load_state_dict(checkpoint['opt'])

    def reset_epoch(self):
        self.trained_epoch = 0
        self.train_loss = []
        self.dev_loss = []

    def train_batch(self, training_batch, tf_rate=1, val=False):
        '''
        train a batch of any batch size

        Inputs:
            training_batch: train data batch created by batch_2_seq
        '''

        # extract fields from batch & set DEVICE options
        image, mask = (i.to(DEVICE) for i in training_batch)

        pred = self.model(image)
        loss = self.loss(pred, mask, image)

        # if in training, not validate
        if not val:
            # Zero gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Adjust model weights
            self.optimizer.step()

        iou = iou_loss(pred, mask)

        return loss.item(), iou.item(), pred

    def train(self, train_data, n_epochs, batch_size=1, stage=None, dev_data=None):
        """
        When we save our model, we save a tarball containing the encoder and decoder state_dicts (parameters),
        the optimizers’ state_dicts, the loss, the iteration, etc.
        After loading a checkpoint, we will be able to use the model parameters to run inference,
        or we can continue training right where we left off.
        """

        start_epoch = self.trained_epoch + 1

        # Data loaders with custom batch builder
        trainloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
            # num_workers=config.LOADER_WORKERS
        )

        self.log(f'Start training from epoch {start_epoch} to {n_epochs}...')

        for epoch in range(start_epoch, n_epochs + 1):
            self.model.train()
            loss_sum, iou_sum = 0, 0

            for idx, training_batch in enumerate(trainloader):
                # run a training iteration with batch
                loss, iou, pred = self.train_batch(training_batch)

                # Accumulate losses to print
                loss_sum += loss
                iou_sum += iou

                # Print progress
                iteration = idx + 1
                if iteration % config.PRINT_EVERY == 0:
                    avg_loss = loss_sum / iteration
                    avg_iou = iou_sum / iteration

                    self.log('Epoch {}; Iter: {}; Percent: {:.1f}%; Avg loss: {:.4f}; Avg IOU: {:.4f};'.format(epoch, iteration, iteration / len(trainloader) * 100, avg_loss, avg_iou))

                    # pred = pred[0]
                    # self.save_sample_imgs(training_batch[0][0], training_batch[1][0], pred, epoch, iteration)


            self.trained_epoch = epoch
            self.train_loss['loss'].append(loss_sum / len(trainloader))
            self.train_loss['iou'].append(iou_sum / len(trainloader))

            if dev_data:
                loss_sum, iou_sum = 0, 0

                self.model.eval()

                devloader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

                for i, dev_batch in enumerate(devloader):
                    loss, iou, pred = self.train_batch(dev_batch, val=True)

                    # Accumulate losses to print
                    loss_sum += loss
                    iou_sum += iou

                    # if i == 0:
                    #     pred = pred[0]
                    #     self.save_sample_imgs(dev_batch[0][0], dev_batch[1][0], pred, epoch, 'val')

                avg_loss = loss_sum / len(devloader)
                avg_iou = iou_sum / len(devloader)

                self.log('Validation; Epoch {}; Avg loss: {:.4f}; Avg IOU: {:.4f};'.format(epoch, avg_loss, avg_iou))

                self.dev_loss['loss'].append(avg_loss)
                self.dev_loss['iou'].append(avg_iou)

            # Save checkpoint
            if epoch % config.SAVE_EVERY == 0:
                cp_name = f'{stage}_{epoch}'
                self.checkpoint_mng.save(cp_name, {
                    'epoch': epoch,
                    'train_loss': self.train_loss,
                    'dev_loss': self.dev_loss,
                    'model': self.model.state_dict(),
                    'opt': self.optimizer.state_dict()
                })

                self.log('Save checkpoint:', cp_name)

    def save_sample_imgs(self, img, mask, prediction, epoch, iter):
        fig = create_figure(img, mask, prediction.float())

        self.checkpoint_mng.save_image(f'{epoch}-{iter}', fig)
        plt.close(fig)
