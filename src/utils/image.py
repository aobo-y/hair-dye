import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def create_figure(img, mask, prediction, dye=False):
  return create_multi_figure([(img, mask, prediction)], dye)

def create_multi_figure(rows, dye=False):
  fig = plt.figure()

  for i, (img, mask, prediction) in enumerate(rows):
    img = (img + 1) / 2

    data = [img, mask, prediction]
    names = ["Image", "Mask", "Prediction"]

    if dye:
      transform_hue = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(hue=0.5, saturation=0.6, contrast=0.2, brightness=0.2),
        transforms.ToTensor()
      ])
      dyed = transform_hue(img)

      dyed = prediction * dyed + (1 - prediction) * img
      data.append(dyed)
      names.append('Dye')


    for j, d in enumerate(data):
      d = d.squeeze()
      im = d.data.cpu().numpy()

      if im.shape[0] != 3:
          im = np.expand_dims(im, axis=0)
          im = np.concatenate((im, im, im), axis=0)

      im = im.transpose(1, 2, 0)

      f = fig.add_subplot(len(rows), len(data), i * len(data)+ j + 1)
      f.imshow(im)
      if i == 0:
        f.set_title(names[j])
      f.set_xticks([])
      f.set_yticks([])

  return fig
