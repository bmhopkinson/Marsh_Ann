import csv
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path, PureWindowsPath

infile = '../Marsh_Explorer_dataset/marsh_explorer_pennings_data_curated.txt'
train_file ='marsh_explorer_pennings_data_split_train.txt'
val_file   ='marsh_explorer_pennings_data_split_val.txt'
test_file  ='marsh_explorer_pennings_data_split_test.txt'
#infile = 'marsh_explorer_pennings_data_trial.txt'

N_CLASSES = 6
N_SECTIONS = 15
VAL_FRAC  = 0.2
TEST_FRAC = 0.2

#with open(infile) as csvfile:
#    data_reader = csv.reader(csvfile,delimiter='\t')
data = pd.read_csv(infile,sep='\t')
fnames = data.iloc[:,5]

start_idx = 7
f_idx = 0
data_out = [];
for f in fnames:
    fwp = PureWindowsPath(f)
    file_base = os.path.splitext(fwp.as_posix())[0]
    for i in range(N_SECTIONS):
        this_file = "./image_sections/" + file_base + "_" + str(i) + ".jpg"
        this_data = []
        for j in range(N_CLASSES):
            this_idx = start_idx + i + j*(N_SECTIONS+1)
            this_data.append(data.iloc[f_idx,this_idx])
        data_out.append([this_file, this_data])

    f_idx = f_idx + 1

#test
#data_out.append(['test',[0, 0, 0, 0, 0, 0]])
#print out data

ftrain = open(train_file,'w')
fval   = open(val_file,'w')
ftest  = open(test_file,'w')
for datum in data_out:
    roll = random.random()
    if roll < TEST_FRAC:
        fout = ftest
    elif( (roll >= TEST_FRAC) and  (roll < (TEST_FRAC + VAL_FRAC))):
        fout  = fval
    else:
        fout = ftrain

    fout.write('%s\t' % datum[0])

    #if no plants were found - code as "empty" image
    if(np.sum(datum[1]) == 0):
        datum[1].append(1)
    else:
        datum[1].append(0)

    for i in range(N_CLASSES+1):
        fout.write('%d\t' % datum[1][i])
    fout.write('\n')
