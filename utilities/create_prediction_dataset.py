import os
import re

base_dir = './image_sections/test/'
img_regex = re.compile('.*\.jpg', re.IGNORECASE)

outfile ='listofimgstopredict.txt'

fout = open(outfile,'w')

for (dirpath, dirnames, files) in os.walk(base_dir, topdown=True):
    for name in files:
        m = img_regex(name)
        if(m):
            fullpath = dirpath + name + '\n'
            fout.write(fullpath)
