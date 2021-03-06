# evaluate performance of CNN classifier
import numpy as np
import cv2
import torch
import torch.nn as nn
from marsh_plant_dataset import MarshPlant_Dataset_pc, MarshPlant_Dataset_pa
from sklearn.metrics import confusion_matrix
import itertools
from torchvision import transforms
import matplotlib.pyplot as plt
from PerformanceMetrics import PerformanceMetrics , PerformanceMetricsPerClass

class Evaluator(object):
    def __init__(self, model_path, data_type="pa", modelname="resnet", batch_size=32,transform=None, config_file = None):
        self.outfile ='performance_bozo_2.txt'
        self.THRESHOLD_SIG = 0.5
        self.batch_size = batch_size
        self.bShuffle = False
        self.num_workers = 8
        self.modelname=modelname
        self.data_type=data_type
        self.transform=transform
        self.data_loader = []
        self.config_file = config_file

        if(data_type=="pa"):
            self.N_CLASSES = 7
            self.Pennings_Classes = ['Salicornia','Spartina','Limonium','Borrichia','Batis','Juncus','None']
        elif(data_type=='pc'):
            self.N_CLASSES = 9
            self.Pennings_Classes = [ 'Spartina','Juncus', 'Salicornia','Batis','Borrichia','Limonium','Soil' ,'other','Unknown' ]

        print('evaluating {}'.format(model_path))
        model = torch.load(model_path)
        #print(model)
        self.model = model.eval()


    def setup_dataloader(self,dataset):
        self.data_loader =torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = self.bShuffle, num_workers = self.num_workers)

    def run(self):

        cpu = torch.device("cpu")
        gpu = torch.device("cuda")

        pred = np.empty((0,self.N_CLASSES), int)
        ann  = np.empty((0,self.N_CLASSES), int)

        sigfunc = nn.Sigmoid()

        metrics = PerformanceMetrics()
        metrics_per_class = PerformanceMetricsPerClass(self.N_CLASSES)

        with torch.no_grad():
            for it, batch in enumerate(self.data_loader):
                input = batch['X'].cuda()#to(device)
                output = self.model(input).to(cpu)

                if(self.data_type=='pa'):
                    sig = output.detach().numpy()
                    this_pred=np.zeros_like(sig)
                    this_pred = sig > self.THRESHOLD_SIG;

                elif(self.data_type=='pc'):
                    sig = sigfunc(output)
                    sig = sig.detach().numpy()
                    this_pred=np.zeros_like(sig)
                    this_pred=(sig == sig.max(axis=1)[:,None]).astype(int)

                #print(this_pred)
                pred = np.append(pred, this_pred.astype(int), axis = 0)
                this_ann = batch['Y'].to(cpu).detach().numpy()  #take off gpu, detach from gradients
                ann = np.append(ann, this_ann.astype(int), axis = 0)

                n_samples =  input.size(0)
                metrics.accumulate(0.0, self.batch_size, this_pred.astype(int), this_ann.astype(int))
                metrics_per_class.accumulate(0.0, self.batch_size, this_pred.astype(int), this_ann.astype(int))

        self.ann=ann
        self.pred=pred

        fout = open(self.outfile,'a')
        fout.write('%s\n'%self.config_file)
        fout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % \
            ("TruePos","TrueNeg","FalsePos","FalseNeg","TruePosRt","TrueNegRt","FalsePosRt","FalseNegRt","Precision", "Recall", "F1"))

        macro_precision=0.0
        macro_recall=0.0
        for i in range(self.N_CLASSES):
            fout.write('%s\t' % self.Pennings_Classes[i])
            fout.write('%d\t%d\t%d\t%d\t' % (metrics_per_class.true_pos[i], metrics_per_class.true_neg[i],metrics_per_class.false_pos[i], metrics_per_class.false_neg[i]) )
            fout.write('%f\t%f\t%f\t%f\t' % (metrics_per_class.true_pos_rate[i], metrics_per_class.true_neg_rate[i],metrics_per_class.false_pos_rate[i], metrics_per_class.false_neg_rate[i]) )
            fout.write('%f\t%f\t%f\t'     % (metrics_per_class.precision[i], metrics_per_class.recall[i],metrics_per_class.f1[i] ) )
            fout.write('\n')
            macro_precision += metrics_per_class.precision[i]
            macro_recall    += metrics_per_class.recall[i]

        fout.write('micro_precision: {:.4f}\t micro_recall: {:.4f}\t micro_f1: {:.4f}\n'.format(metrics.precision, metrics.recall, metrics.f1))

        macro_precision = macro_precision/self.N_CLASSES
        macro_recall    = macro_recall   /self.N_CLASSES
        macro_f1= (2*macro_precision * macro_recall)/(macro_precision + macro_recall)
        fout.write('macro_precision: {:.4f}\t macro_recall: {:.4f}\t macro_f1: {:.4f}\n'.format(macro_precision, macro_recall, macro_f1))
        fout.write('______________________________\n')
        fout.close()

if __name__ == "__main__":
    model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn92', pretrained=True)

    image_dim=(512,512)
    modellist=['dpn','densenet','resnext','resnet','inception','pyramid']
    for modelname in modellist:
        if modelname=='pyramid':
            image_dim=(224,224)
        transform_test = transforms.Compose([
        transforms.Resize(image_dim),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        #performer=Performer(data_type="pa",modelname="ResNet101_row50_2020",transform=transform_test)
        performer=Evaluator(data_type="pa",modelname=modelname,transform=transform_test)
