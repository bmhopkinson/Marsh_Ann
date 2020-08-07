import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL
import yaml

from marsh_plant_dataset import MarshPlant_Dataset_pa, MarshPlant_Dataset_pc


def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
  plt.imshow(img)
  plt.axis('off')
  plt.show()

image_dim=(512,512)
crop_dim = (1000,1000)
datafiles = ['small_pa_sample.txt']

transforms_base = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(crop_dim),
    transforms.Resize(image_dim),
    #transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transforms_fancy = transforms.Compose([
    transforms.ColorJitter(hue=.02, saturation=.02),
    transforms.RandomAffine(20, translate=None, scale = (0.8, 1.1), shear = 10,
        resample = PIL.Image.BILINEAR, fillcolor=0)
    ])

transforms_test = transforms.Compose([
    transforms_fancy,
    transforms_base,
    ])

#train_data = MarshPlant_Dataset_pa(datafiles,transform=transforms)

mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

dataset = torchvision.datasets.ImageFolder('/home/cv-bhlab/Documents/Marsh_Ann/Jayant/PA_PC/image_sections/trial/', transform=transforms_test)
show_dataset(dataset)
