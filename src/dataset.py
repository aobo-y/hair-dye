import os
import random

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class ImgTransformer():
    def __init__(self, img_size, color_aug=False):
        self.img_size = img_size
        self.color_aug = color_aug

    def transform(self, image):
        transformer = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.img_size)
        ] + ([
            transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=.5),
            transforms.RandomGrayscale(p=0.1)
        ] if self.color_aug else []) + [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        transform_image = transformer(image)

        return transform_image

    def load(self, path):
        image = Image.open(path).convert('RGB')
        return self.transform(image)



class HairDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, image_size=448, color_aug=False):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception("%s  not exists." % self.data_folder)

        self.imagedir_path = os.path.join(data_folder, "images")
        self.maskdir_path = os.path.join(data_folder, "masks")
        self.image_names = os.listdir(self.imagedir_path)

        self.image_size = image_size

        self.transformer = ImgTransformer(image_size, color_aug)

    def __getitem__(self, index):
        img_path = os.path.join(self.imagedir_path, self.image_names[index])
        transform_image = self.transformer.load(img_path)

        maskfilename = self.image_names[index].split('.')[0] + '.pbm'
        mask = Image.open(os.path.join(self.maskdir_path, maskfilename))

        transform_mask = transforms.Compose([
            transforms.CenterCrop(min(mask.size[0], mask.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])(mask)

        # random horizontal flip
        hflip = random.choice([True, False])
        if hflip:
            transform_image = transform_image.flip([2])
            transform_mask = transform_mask.flip([2])

        return transform_image, transform_mask

    def __len__(self):
        return len(self.image_names)

