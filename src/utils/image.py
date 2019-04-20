import numpy as np
import matplotlib.pyplot as plt

def create_figure(img, mask, prediction):
  data = [img, mask, prediction]
  names = ["Image", "Mask", "Prediction"]

  fig = plt.figure()
  for i, d in enumerate(data):
    d = d.squeeze()
    im = d.data.cpu().numpy()

    if i > 0:
        im = np.expand_dims(im, axis=0)
        im = np.concatenate((im, im, im), axis=0)
    else:
        im = (im + 1) / 2

    im = im.transpose(1, 2, 0)

    f = fig.add_subplot(1, 3, i + 1)
    f.imshow(im)
    f.set_title(names[i])
    f.set_xticks([])
    f.set_yticks([])

  return fig
