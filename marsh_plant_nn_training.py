import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.distributed as dist
import torchvision
from torchvision import transforms
import PIL

import Evaluator
import Trainer
from marsh_plant_dataset import MarshPlant_Dataset_pa, MarshPlant_Dataset_pc


import pdb


if __name__ == "__main__":
	print("PyTorch Version: ",torch.__version__)
	#inputs
	data_type="pa"
	modellist= ['resnext']  #['dpn']#,'neat' 'aawide','resnext','densenet','resnet','inception','pyramid','dpn']
	image_dim=(512,512)
	crop_dim = (1000,1000)

	datafiles  = { 'pa':
			{ 'train' : ['./infiles/pa_2014_ann_train.txt', './infiles/pa_2014_spartadd_train.txt', './infiles/pa_2014_juncadd_train.txt'], #['small_pa_sample.txt'],
		  	  'val'   : ['./infiles/pa_2014_ann_val.txt'  , './infiles/pa_2014_spartadd_val.txt'  , './infiles/pa_2014_juncadd_val.txt'],
			  'test'  : ['./infiles/pa_2014_ann_test.txt' , './infiles/pa_2014_spartadd_test.txt'  , './infiles/pa_2014_juncadd_test.txt']
		},

		'pc':
	 		{ 'train' : 'marsh_percent_cover_train.txt',
	 		  'val'   : 'marsh_percent_cover_val.txt',
			  'test'   : 'marsh_percent_cover_val.txt'
	 		}

	 }

	train_params = {
		'batch_size_top' : 16 ,
		'batch_size_all' : 4 ,
		'epochs_top' : 3 ,
		'epochs_all' : 3
	 }

	distributed=False

	for modelname in modellist:  #convert this to loop on info specified in config file

		trainer = Trainer.Trainer(train_params, data_type=data_type,modelname=modelname)
		if(modelname=="pyracrop_dim = (1000,1000)mid" ): #or modelname=='dpn'):
			image_dim=(224,224)
		if(modelname=="aawide" ):
			image_dim=(32,32)
		#if(modelname=="dpn"):
			#distributed=True
		if distributed :
			dist.init_process_group(backend='gloo', init_method='env://',
                                rank=0,world_size=4)

        #setup data transforms
		transforms_base = transforms.Compose([
	    	transforms.Resize(image_dim),
	    	transforms.ToTensor(),
	    	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		transform_train = transforms.Compose([
			transforms.RandomVerticalFlip(),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(hue=.02, saturation=.02),
			transforms.RandomAffine(20, translate=None, scale = (0.8, 1.1), shear = 10,
				resample = PIL.Image.BILINEAR, fillcolor=0),
			transforms.CenterCrop(crop_dim),
			transforms_base
		])

		transform_test = transforms_base;
		transform_val = transforms_base;

		#load datasets and setup dataloaders
		if(data_type=="pa"):
			train_data = MarshPlant_Dataset_pa(datafiles[data_type]['train'] ,transform=transform_train)
			val_data   = MarshPlant_Dataset_pa( datafiles[data_type]['val']  ,transform=transform_val)
			test_data  = MarshPlant_Dataset_pa( datafiles[data_type]['test'] ,transform=transform_test)
			criterion = nn.BCEWithLogitsLoss().cuda()
		if(data_type=="pc"):
			train_data = MarshPlant_Dataset_pc(datafiles[data_type]['train'],transform=transform_train)
			val_data  = MarshPlant_Dataset_pc( datafiles[data_type]['val']  ,transform=transform_val)
			test_data  = MarshPlant_Dataset_pc( datafiles[data_type]['test'] ,transform=transform_test)
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
		performer.setup_dataloader(test_data)
		performer.run()
		print("Finished Performer class on test")
