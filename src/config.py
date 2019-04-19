# -*- coding: utf-8 -*-

# Corpus, path relateds to /src
TRAIN_CORPUS = '../data/dataset_figaro1k/training'
TEST_CORPUS = '../data/dataset_figaro1k/testing'

# checkpoints relevant
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'default'
SAVE_EVERY = 4       # save the checkpoint every x epochs

# Epochs of training
EPOCHS = 60

# Configure models - training relevant

# Configure training/optimization
LOADER_WORKERS = 4
BATCH_SIZE = 4            # size of the mini batch in training state
LR = 0.00004                # learning ratio
PRINT_EVERY = 1000          # print the loss every x iterations

