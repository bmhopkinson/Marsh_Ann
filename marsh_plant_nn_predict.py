import numpy as np
import cv2
import re
import torch
import torch.nn as nn
from torchvision import transforms

from marsh_plant_dataset import MarshPlant_Dataset_pa

N_CLASSES = 7
THRESHOLD_SIG = 0.5
batch_size = 32
bShuffle = False
num_workers = 8
image_dim=(512,512)

id_regex = re.compile('.*Row(\d+).*DSC_(\d+)_(\d+)')
remove_brackets = re.compile('\[(.*)\]')

model_path = './modeling/saved_models/resnext_pa_sig_0.50_20200720.torch'
data_infile  = ['./infiles/2014_Rows51to80_pred_infile.txt']
outfile = '2014_Rows51to80_predictions.txt'

model = torch.load(model_path)
model.eval()
sigfunc = nn.Sigmoid()


transforms_base = transforms.Compose([
    transforms.Resize(image_dim),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


pred_data  = MarshPlant_Dataset_pa(data_infile, train=False, transform = transforms_base)
data_loader = torch.utils.data.DataLoader(pred_data, batch_size = batch_size, shuffle = bShuffle, num_workers = num_workers)

cpu = torch.device("cpu")
gpu = torch.device("cuda")

results = {'Row': [], 'img': [], 'sector' :[], 'pred'  : np.empty((0,N_CLASSES), int) }

with torch.no_grad():
    for it, batch in enumerate(data_loader):
        output = model(batch['X'].to(gpu)).to(cpu)

        sig = sigfunc(output)
        sig = sig.detach().numpy()
        this_pred = sig > THRESHOLD_SIG;
#        print(this_pred.shape)
        results['pred'] = np.append(results['pred'], this_pred.astype(int), axis = 0)

        for file in batch['fname']:
            m = id_regex.search(file)
            if(m):
                #print('Row: {}, img {}, sector {}'.format(m.group(1), m.group(2), m.group(3) ) )
                results['Row'].append(m.group(1))
                results['img'].append(m.group(2))
                results['sector'].append(m.group(3))
            else:
                results['Row'].append('x')
                results['img'].append('x')
                results['sector'].append('x')
        #print(pred.shape)

fout = open(outfile,'w')
for i in range(len(results['Row'])):
    fout.write('{}\t{}\t{}\t'.format(results['Row'][i],results['img'][i],results['sector'][i] ) )
    str_out = np.array2string(results['pred'][i,:])
    m = remove_brackets.match(str_out)
    str_out = m[1]
    fout.write('%s\t' % str_out)
    fout.write('\n')
