from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize
import torch
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import config

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

class HairMattingLoss(nn.Module):
  def __init__(self, ratio_of_Gradient=0.0):
    super(HairMattingLoss, self).__init__()

    self.ratio_of_gradient = ratio_of_Gradient
    self.bce_loss = nn.BCELoss()

    if self.ratio_of_gradient > 0:
      sobel_kernel_x = torch.Tensor([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
      ]).view(1,1,3,3)
      self.sobel_kernel_x = nn.Parameter(sobel_kernel_x, False)

      sobel_kernel_y = torch.Tensor([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]
      ]).view(1,1,3,3)
      self.sobel_kernel_y = nn.Parameter(sobel_kernel_y, False)

      rgb_ratios = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
      self.rgb_ratios = nn.Parameter(rgb_ratios, False)


  def forward(self, pred, mask, img):
    loss = self.bce_loss(pred, mask)

    if self.ratio_of_gradient > 0:
      # cnvt to grayscale & keep the channel dim
      grayscale = (img * self.rgb_ratios).sum(1, keepdim=True)

      I_x = F.conv2d(grayscale, self.sobel_kernel_x) / 4
      G_x = F.conv2d(pred, self.sobel_kernel_x) / 4

      I_y = F.conv2d(grayscale, self.sobel_kernel_y) / 4
      G_y = F.conv2d(pred, self.sobel_kernel_y) / 4

      # avoid 0 sqrt
      G = torch.sqrt(G_x.pow(2) + G_y.pow(2) + 1e-6)

      rang_grad = 1 - (I_x * G_x + I_y * G_y).pow(2)

      loss2 = (G * rang_grad).sum((1, 2, 3)) / G.sum((1, 2, 3))
      loss2 = loss2.mean()

      loss = loss + loss2 * self.ratio_of_gradient

    return loss


# def image_gradient(image):
#   edges_x = filters.sobel_h(image)
#   edges_y = filters.sobel_v(image)
#   edges_x = normalize(edges_x)
#   edges_y = normalize(edges_y)
#   return torch.from_numpy(edges_x), torch.from_numpy(edges_y)


# def image_gradient_loss(image, pred):
#   loss = 0
#   for i in range(len(image)):
#     pred_grad_x, pred_grad_y = image_gradient(pred[i][0].cpu().detach().numpy())
#     gray_image = torch.from_numpy(rgb2gray(image[i].permute(1, 2, 0).cpu().numpy()))
#     image_grad_x, image_grad_y = image_gradient(gray_image)
#     IMx = (image_grad_x * pred_grad_x).float()
#     IMy = (image_grad_y * pred_grad_y).float()
#     Mmag = (pred_grad_x.pow(2) + pred_grad_y.pow(2)).sqrt().float()
#     IM = 1 - (IMx + IMy).pow(2)
#     numerator = torch.sum(Mmag * IM)
#     denominator = torch.sum(Mmag)
#     loss = loss + numerator / denominator

#   return torch.div(loss, len(image))


# def hairmat_loss(pred, image, mask):
#   pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
#   mask_flat = mask.squeeze(1).view(-1).long()
#   cross_entropy_loss = F.cross_entropy(pred_flat, mask_flat)
#   # image_loss = image_gradient_loss(image, pred).to(DEVICE).float()
#   image_loss = torch.tensor(0).float()
#   return cross_entropy_loss, image_loss

def iou_loss(pred, mask):
  pred[pred > 0.5] = 1
  pred[pred <= 0.5] = 0
  pred = pred.squeeze().long()
  mask = mask.squeeze().long()
  Union = torch.where(pred > mask, pred, mask)
  Overlep = torch.mul(pred, mask)
  loss = torch.div(torch.sum(Overlep).float(), torch.sum(Union).float())
  return loss

def acc_loss(pred, mask):
  pred[pred > 0.5] = 1
  pred[pred <= 0.5] = 0
  pred = pred.squeeze().long()
  mask = mask.squeeze().long()
  all_ones = torch.ones_like(mask)
  all_zeros = torch.zeros_like(mask)
  Right = torch.where(pred == mask, all_ones, all_zeros)
  #Overlep = torch.mul(pred, mask)
  loss = torch.div(torch.sum(Right).float(), torch.sum(all_ones).float())
  return loss

def F1_loss(pred, mask):
  pred[pred > 0.5] = 1
  pred[pred <= 0.5] = 0
  pred = pred.squeeze().long()
  mask = mask.squeeze().long()
  all_ones = torch.ones_like(mask)
  all_zeros = torch.zeros_like(mask)
  #Right = torch.where(pred == mask, all_ones, all_zeros)
  Overlep = torch.mul(pred, mask)
  precision = torch.div(torch.sum(Overlep).float(), torch.sum(pred).float())
  recall = torch.div(torch.sum(Overlep).float(), torch.sum(mask).float())
  F1loss = torch.div(torch.mul(precision, recall), torch.add(precision, recall))
  return F1loss*2
