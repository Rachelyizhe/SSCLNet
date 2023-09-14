from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import pandas as pd
import numpy as np
import torch
import os
import random
import glob

import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f


def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2

def tensor_to_pil(im):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    return im


# --------------------------------------------------------------------------------
# Define data augmentation
# --------------------------------------------------------------------------------
def transform(img1, img2):


    rand = random.random()
    if rand < 1 / 6:
        img1 = img1.filter(ImageFilter.BLUR)
        img2 = img2.filter(ImageFilter.BLUR)

    elif rand < 2 / 6:
        img1 = img1.filter(ImageFilter.DETAIL)
        img2 = img2.filter(ImageFilter.DETAIL)

    elif rand < 3 / 6:
        img1 = img1.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img2 = img2.filter(ImageFilter.EDGE_ENHANCE_MORE)

    elif rand < 4 / 6:
        img1 = img1.filter(ImageFilter.SMOOTH)
        img2 = img2.filter(ImageFilter.SMOOTH)

    elif rand < 5 / 6:
        img1 = img1.filter(ImageFilter.SHARPEN)
        img2 = img2.filter(ImageFilter.SHARPEN)


    rand = random.random()
    if rand < 1 / 7:
        img1 = ImageEnhance.Brightness(img1).enhance(factor=1.5)
        img2 = ImageEnhance.Brightness(img2).enhance(factor=1.5)

    elif rand < 2 / 7:
        img1 = ImageEnhance.Brightness(img1).enhance(factor=0.5)
        img2 = ImageEnhance.Brightness(img2).enhance(factor=0.5)

    elif rand < 3 / 7:
        img1 = ImageEnhance.Color(img1).enhance(factor=0.5)
        img2 = ImageEnhance.Color(img2).enhance(factor=0.5)

    elif rand < 4 / 7:
        img1 = ImageEnhance.Color(img1).enhance(factor=1.5)
        img2 = ImageEnhance.Color(img2).enhance(factor=1.5)

    elif rand < 5 / 7:
        img1 = ImageEnhance.Contrast(img1).enhance(factor=0.5)
        img2 = ImageEnhance.Contrast(img2).enhance(factor=0.5)

    elif rand < 6 / 7:
        img1 = ImageEnhance.Contrast(img1).enhance(factor=1.5)
        img2 = ImageEnhance.Contrast(img2).enhance(factor=1.5)


    # Transform to tensor
    image1 = transforms_f.to_tensor(img1)
    image2 = transforms_f.to_tensor(img2)

    # Apply (ImageNet) normalisation
    image1 = transforms_f.normalize(image1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image2 = transforms_f.normalize(image2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image1, image2
 
 

def batch_transform(data1, data2):
    data_list1 = []
    device1 = data1.device
    data_list2 = []
    device2 = data2.device

    for k in range(data1.shape[0]):
        data_pil1 = tensor_to_pil(data1[k])
        data_pil2 = tensor_to_pil(data2[k])
        aug_data1, aug_data2 = transform(data_pil1, data_pil2)
        data_list1.append(aug_data1.unsqueeze(0))
        data_list2.append(aug_data2.unsqueeze(0))

    data_trans1 = torch.cat(data_list1).to(device1)
    data_trans2 = torch.cat(data_list2).to(device2)

    return data_trans1, data_trans2




