import os
import fnmatch
import re
import numpy as np
import random

topdir = '../Marsh_Annotations_BH'
train_file = 'BH_ann_train.txt'
val_file   = 'BH_ann_val.txt'
test_file  = 'BH_ann_test.txt'
class_regex = re.compile('(\d+)\t(\w*)')
strip_regex = re.compile('../(.*)_ann.txt$')
Pennings_Classes = ['Sarcicornia','Spartina','Limonium','Borrichia','Batis','Juncus']

N_SECTIONS = 15
VAL_FRAC  = 0.2
TEST_FRAC = 0.2

def read_BHann(file):  #data file reader
    anns = {}
    fid = open(file,'r')
    line = fid.readline()
    while(line):
        m=class_regex.match(line)  #new class
        if(m):
            class_name = m.group(2)
            this_anns = [];
            for i in range(3): #3 lines of data_out
               dline = fid.readline().rstrip()
               this_anns.extend(list(dline))

            anns[class_name] = this_anns
        line =fid.readline()
    return anns

def ordered_datum(img_dict,sector):
    out_data = []
    for c in Pennings_Classes:
        this_class_datum = img_dict[c][sector]
        #print(this_class_datum)
        out_data.append(this_class_datum)

    out_data = list(map(int,out_data))
    if(np.sum(out_data) == 0):
        out_data.append(1)
    else:
        out_data.append(0)

    return out_data

## read in data from file
all_data = {}
for (dirpath, dirname, files) in os.walk(topdir, topdown='True'):
    for name in files:
        if fnmatch.fnmatch(name,'*.txt'):
            file_path = os.path.join(dirpath,name)
            this_data = read_BHann(file_path)
            all_data[file_path] = this_data

#write out in format for machine learning
ftrain = open(train_file,'w')
fval   = open(val_file,'w')
ftest  = open(test_file,'w')
for x in all_data:
    m = strip_regex.search(x)
    for i in range(N_SECTIONS):
        roll = random.random()
        if roll < TEST_FRAC:
            fout = ftest
        elif( (roll >= TEST_FRAC) and  (roll < (TEST_FRAC + VAL_FRAC))):
            fout  = fval
        else:
            fout = ftrain

        this_file = "./image_sections/" + m[1] + "_" + str(i) + ".jpg"
        this_datum = ordered_datum(all_data[x],i)
        fout.write('%s\t' % this_file)
        fout.writelines(['%d\t' % item for item in this_datum])
        fout.write('\n')

ftrain.close()
fval.close()
ftest.close()
