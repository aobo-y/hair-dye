"""
SouthPark Chatbot
"""

import os
import argparse
import torch

import config
from models import MobileHairNet
from trainer import Trainer
from evaluate import test
from dataset import HairDataset

from utils import CheckpointManager

DIR_PATH = os.path.dirname(__file__)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

SAVE_PATH = os.path.join(DIR_PATH, config.SAVE_DIR, config.MODEL_NAME)


def build_model(checkpoint):
    model = MobileHairNet()

    if checkpoint:
        model.load_state_dict(checkpoint['model'])

    # Use appropriate device
    model = model.to(device)

    return model


def train(mode, model, checkpoint, checkpoint_mng):
    trainer = Trainer(model, checkpoint_mng)

    if checkpoint:
        trainer.resume(checkpoint)

    trianfile = os.path.join(DIR_PATH, config.TRAIN_CORPUS)
    devfile = os.path.join(DIR_PATH, config.TEST_CORPUS)

    print("Reading training data from %s..." % trianfile)

    train_datasets = HairDataset(trianfile, 224)

    print(f'Read {len(train_datasets)} training images')

    print("Reading development data from %s..." % devfile)

    dev_datasets = HairDataset(devfile, 224)

    print(f'Read {len(dev_datasets)} development images')

    # Ensure dropout layers are in train mode
    model.train()

    trainer.train(train_datasets, config.EPOCHS, config.BATCH_SIZE, stage=mode, dev_data=dev_datasets)

def evaluate(model, checkpoint):
    # Set dropout layers to eval mode
    model.eval()

    testfile = os.path.join(DIR_PATH, config.TEST_CORPUS)
    print("Reading Testing data from %s..." % testfile)

    test_datasets = HairDataset(testfile)

    print(f'Read {len(test_datasets)} testing images')

    test(test_datasets, model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices={'train', 'test'}, help="mode to run the network")
    parser.add_argument('-cp', '--checkpoint')
    args = parser.parse_args()

    print('Saving path:', SAVE_PATH)
    checkpoint_mng = CheckpointManager(SAVE_PATH)

    checkpoint = None
    if args.checkpoint:
        print('Load checkpoint:', args.checkpoint)
        checkpoint = checkpoint_mng.load(args.checkpoint, device)

    model = build_model(checkpoint)

    if args.mode == 'train':
        train(args.mode, model, checkpoint, checkpoint_mng)

    elif args.mode == 'test':
        evaluate(model, checkpoint)


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--checkpoint')
    args = parser.parse_args()

    checkpoint_mng = CheckpointManager(SAVE_PATH)
    checkpoint = None if not args.checkpoint else checkpoint_mng.load(args.checkpoint, device)

    model = build_model(checkpoint)
    # Set dropout layers to eval mode
    model.eval()

    return model


if __name__ == '__main__':
    main()
