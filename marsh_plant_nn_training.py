import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import Evaluator
import Trainer

from marsh_plant_dataset import MarshPlant_Dataset_pa, MarshPlant_Dataset_pc
import torch.utils.data.distributed
import torch.distributed as dist

import pdb

train_infile = ['./infiles/pa_2014_ann_train.txt']#, './infiles/pa_2014_spartadd_train.txt', './infiles/pa_2014_juncadd_train.txt']
val_infile   = ['./infiles/pa_2014_ann_val.txt'  ]#, './infiles/pa_2014_spartadd_val.txt'  , './infiles/pa_2014_juncadd_val.txt']

if __name__ == "__main__":
	print("PyTorch Version: ",torch.__version__)
	#inputs
	data_type="pa"
	modellist= ['resnext']  #['dpn']#,'neat' 'aawide','resnext','densenet','resnet','inception','pyramid','dpn']
	image_dim=(512,512)
	datafiles  = { 'pa':
			{ 'train' : ['./infiles/pa_2014_ann_train.txt', './infiles/pa_2014_spartadd_train.txt', './infiles/pa_2014_juncadd_train.txt'],
		  	  'val'   : ['./infiles/pa_2014_ann_val.txt'  , './infiles/pa_2014_spartadd_val.txt'  , './infiles/pa_2014_juncadd_val.txt']
		},

		'pc':
	 		{ 'train' : 'marsh_percent_cover_train.txt',
	 		  'val'   : 'marsh_percent_cover_val.txt'
	 		}

	 }

	train_params = {
		'batch_size_top' : 16 ,
		'batch_size_all' : 4 ,
		'epochs_top' : 3 ,
		'epochs_all' : 3
	 }

	distributed=False

	for modelname in modellist:

		trainer = Trainer.Trainer(train_params, data_type=data_type,modelname=modelname)
		if(modelname=="pyramid" ): #or modelname=='dpn'):
			image_dim=(224,224)
		if(modelname=="aawide" ):
			image_dim=(32,32)
		#if(modelname=="dpn"):
			#distributed=True
		if distributed :
			dist.init_process_group(backend='gloo', init_method='env://',
                                rank=0,world_size=4)

        #setup data transforms
		transform_train = transforms.Compose([
		transforms.Resize(image_dim),transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

		transform_test = transforms.Compose([
	    	transforms.Resize(image_dim),
	    	transforms.ToTensor(),
	    	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		transform_val = transforms.Compose([
	    	transforms.Resize(image_dim),
	    	transforms.ToTensor(),
	    	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		#load datasets and setup dataloaders
		if(data_type=="pa"):
			train_data = MarshPlant_Dataset_pa(datafiles[data_type]['train'],transform=transform_train)
			val_data  = MarshPlant_Dataset_pa( datafiles[data_type]['val']  ,transform=transform_val)
			criterion = nn.BCEWithLogitsLoss().cuda()
		if(data_type=="pc"):
			train_data = MarshPlant_Dataset_pc(datafiles[data_type]['train'],transform=transform_train)
			val_data  = MarshPlant_Dataset_pc( datafiles[data_type]['val']  ,transform=transform_val)
			criterion = nn.CrossEntropyLoss().cuda()#change to cross entropy here.
		datasets = {'train' : train_data, 'val' : val_data}

		if distributed:
			train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
			val_sampler   = torch.utils.data.distributed.DistributedSampler(val_data  )
			samplers={'train' : train_sampler, 'val' : val_sampler}

		bShuffle = True
		num_workers = 8
		if not distributed:
			samplers=None

		trainer.setup_dataloaders(datasets, bShuffle, num_workers,samplers=samplers)
		trainer.setup_model(distributed) #setup model and training parameters

		best_f1 = 0.0
		gamma = {'top' :0.5, 'all': 0.8}
		for stage in ['top', 'all']:
		# Train Top (fc) layer
			trainer.set_optimizable_parameters(stage)
			print("Training {}: {} parameters optimized".format(stage, trainer.count_optimizable_parameters() ) )

			optimizer = torch.optim.Adam(trainer.optimizable_parameters, lr = trainer.lr[stage])
			lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma[stage])

			best_f1 = trainer.train(stage, criterion, optimizer, scheduler = lr_scheduler, best_score=best_f1)
			print('Finished training {}, best acc {:.4f}'.format(stage, best_f1))

		performer=Evaluator.Evaluator(data_type=data_type,modelname=modelname,transform=transform_test)
		print("Finished Performer class on test")
