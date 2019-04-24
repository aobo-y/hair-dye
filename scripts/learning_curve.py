import os
import torch
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(__file__)

cp_path = os.path.join(DIR_PATH, '../src/checkpoints')

file_path = os.path.join(cp_path, 'default/train_100.tar')


def get_loss(filepath):
  checkpoint = torch.load(filepath, map_location='cpu')
  return checkpoint['train_loss'], checkpoint['dev_loss']


def val_loss():
  loss = plt.subplot(121)
  iou = plt.subplot(122)
  loss.grid()
  iou.grid()
  loss.set_title('Loss')
  iou.set_title('IOU')

  train_loss, dev_loss = get_loss(file_path)

  loss.plot(train_loss['loss'], label='train')
  loss.plot(dev_loss['loss'], label='test')
  iou.plot(train_loss['iou'], label='train')
  iou.plot(dev_loss['iou'], label='test')

  loss.legend(loc='best')
  iou.legend(loc='best')

  plt.show()

val_loss()
