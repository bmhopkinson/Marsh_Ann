import os
import cv2
import numpy as np
import re

path_regex = re.compile('^.+?/(.*)')

def resize_image(im, factor):
    row, col, chan = im.shape
    col_re = np.rint(col*factor).astype(int)
    row_re = np.rint(row*factor).astype(int)
    im = cv2.resize(im, (col_re, row_re)) #resize patch

    return im


imdir = '../Marsh_Images_BH/Row1_1_2748to2797'
outdir = './image_resize_BH'
for (dirpath, dirname, files) in os.walk(imdir, topdown='True'):
    for name in files:
        fullpath = os.path.join(dirpath,name)
        print(name)

        m = path_regex.findall(dirpath)
        dirpath_sub = m[0]
        new_dirpath = os.path.join(outdir,dirpath_sub)
        if not os.path.isdir(new_dirpath):
            os.makedirs(new_dirpath)

        file_base = os.path.splitext(name)[0]
        im = cv2.imread(fullpath)
        im_alt = resize_image(im, 0.2)
        outfile = file_base + '_small.jpg'
        outpath = os.path.join(new_dirpath, outfile)
        cv2.imwrite(outpath,im_alt)
