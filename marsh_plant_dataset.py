
import numpy as np
import cv2
import csv
import multiprocessing

import torch
from torch.utils.data import Dataset
from PIL import Image

def try_image(img_file):
    result = []
    try:
        im = cv2.imread(img_file)
        height, width = im.shape[:2]
        if height > 100 and width > 100:
            result = img_file
    except:
        print("{} does not refer to a valid img file".format(img_file))
    return result

class MarshPlant_Dataset_pa(Dataset):
    def __init__(self, infiles, train = True, transform=None):
        #initialize dataset
        self.imgfiles = []
        self.anns = []
        self.transform= transform
        self.train = train

        for file in infiles:
            imgs_to_validate = {}
            with open(file,'r') as f:
                reader = csv.reader(f,delimiter='\t')
                for row in reader:
                    fname = row[0]
                    if(self.train):
                        anns = list(map(int,row[1:8]))
                        imgs_to_validate[fname] = anns
                    else:
                        imgs_to_validate[fname] = []

                imgs_validated = []

                pool = multiprocessing.Pool(processes = 8)
                imgs_validated = pool.map(try_image, imgs_to_validate.keys())
                imgs_validated = [x for x in imgs_validated if x !=[]]  #remove empty lists indicated image was not valid
                cleaned = dict((k, imgs_to_validate[k]) for k in imgs_validated)

                files_list = list(cleaned.keys())
                anns_list = list(cleaned.values())
                self.imgfiles.extend(files_list)
                self.anns.extend(anns_list)

    def __len__(self):
        #return length of dataset
        return len(self.imgfiles)

    def __getitem__(self, idx):
        im = Image.open(self.imgfiles[idx])
        x = self.transform(im)
        if self.train:
            y = torch.tensor(self.anns[idx]).float()  #annotations
            return {'X': x, 'Y': y}
        else:
            return {'X': x, 'fname': self.imgfiles[idx]}

class MarshPlant_Dataset_pc(Dataset):
    def __init__(self, infile,transform=None):
        #initialize dataset
        self.imgfiles = []
        self.anns = []
        self.transform= transform
        with open(infile,'r') as f:
            reader = csv.reader(f,delimiter='\t')
            for row in reader:
                #print(row[0])
                self.imgfiles.append(row[0])
                ann = list(map(int,row[1:10])) # change to 1:8 for pa and 1:10 for percent cover
                self.anns.append(ann)

    def __len__(self):
        #return length of dataset
        return len(self.imgfiles)

    def __getitem__(self, idx):
        #return dataset[idx]
        #print(self.imgfiles[idx])
        y = torch.tensor(self.anns[idx]).float()  #annotations
        im = Image.open(self.imgfiles[idx])
        #im = cv2.imread(self.imgfiles[idx])
        #dst = cv2.fastNlMeansDenoisingColored(im,None,10,10,7,21)
        #im = cv2.resize(im, (512,512)) #resize patch
        #mean_value= im.mean()
        #subtracting each pixel of the image from mean
        #im= mean_value - im
        #im = np.transpose(im,(2,0,1))/ 255.
        #im = np.expand_dims(im, axis=0)
        #denoising instead of image mean centering
        #x = torch.from_numpy(im).float()
        x = self.transform(im)
        return {'X': x, 'Y': y}
