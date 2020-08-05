import os
import re

base_dir = './image_sections/2014'
img_regex = re.compile('^(.*)\.[jJ][pP][eE]?[gG]')
id_regex = re.compile('.*Row(\d+).*DSC_(\d+)_(\d+)')
#id_regex = re.compile('DSC_(\d+)_(\d+)')
outfile ='2014_Rows21to50_pred_infile.txt'

fout = open(outfile,'w')

for (dirpath, dirnames, files) in os.walk(base_dir, topdown=True):
    for name in files:
        print(name)
        m = img_regex.search(name)
        if(m):
            fullpath = dirpath + '/' + name + '\n'
            fout.write(fullpath)

#            m2 = id_regex.search(fullpath)
#            if(m2):
#                print('Row: {}, img {}, sector {}'.format(m2.group(1), m2.group(2), m2.group(3) ) )
