from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize
import torch

from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import config

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def image_gradient(image):
  edges_x = filters.sobel_h(image)
  edges_y = filters.sobel_v(image)
  edges_x = normalize(edges_x)
  edges_y = normalize(edges_y)
  return torch.from_numpy(edges_x), torch.from_numpy(edges_y)


def image_gradient_loss(image, pred):
  loss = 0
  for i in range(len(image)):
    pred_grad_x, pred_grad_y = image_gradient(pred[i][0].cpu().detach().numpy())
    gray_image = torch.from_numpy(rgb2gray(image[i].permute(1, 2, 0).cpu().numpy()))
    image_grad_x, image_grad_y = image_gradient(gray_image)
    IMx = (image_grad_x * pred_grad_x).float()
    IMy = (image_grad_y * pred_grad_y).float()
    Mmag = (pred_grad_x.pow(2) + pred_grad_y.pow(2)).sqrt().float()
    IM = 1 - (IMx + IMy).pow(2)
    numerator = torch.sum(Mmag * IM)
    denominator = torch.sum(Mmag)
    loss = loss + numerator / denominator

  return torch.div(loss, len(image))


def hairmat_loss(pred, image, mask):
  pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
  mask_flat = mask.squeeze(1).view(-1).long()
  cross_entropy_loss = F.cross_entropy(pred_flat, mask_flat)
  image_loss = image_gradient_loss(image, pred).to(DEVICE)
  return cross_entropy_loss, image_loss.float()

def iou_loss(pred, mask):
  pred = torch.argmax(pred, 1).long()
  mask = torch.squeeze(mask).long()
  Union = torch.where(pred > mask, pred, mask)
  Overlep = torch.mul(pred, mask)
  loss = torch.div(torch.sum(Overlep).float(), torch.sum(Union).float())
  return loss
