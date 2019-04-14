# -*- coding: utf-8 -*-

# Corpus, path relateds to /src
TRAIN_CORPUS = 'data/output/split/train/...'
DEV_CORPUS = 'data/output/split/dev/...'
TEST_CORPUS = 'data/output/split/test/...'

# checkpoints relevant
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'default'
SAVE_EVERY = 4       # save the checkpoint every x epochs

# Epochs of training
EPOCHS = 60

# Configure models - training relevant

# Configure training/optimization
BATCH_SIZE = 16            # size of the mini batch in training state
LR = 0.00004                # learning ratio
PRINT_EVERY = 1000          # print the loss every x iterations

