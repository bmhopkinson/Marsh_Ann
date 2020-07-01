import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from modeling.backbone.resnet import ResNet101
from marsh_plant_dataset import MarshPlant_Dataset_pa, MarshPlant_Dataset_pc
from RN101_newtop import RN101_newtop
import Evaluator
import Trainer
#import PyramidNet_model.PyramidNet as prn
import torch.utils.data.distributed
import torch.distributed as dist
#from Attention_Augmented.attention_augmented_wide_resnet import Wide_ResNet
#from neatcnn import NeatCNN


if __name__ == "__main__":
	print("PyTorch Version: ",torch.__version__)
	#inputs
	data_type="pa"
	modellist= ['resnet']  #['dpn']#,'neat' 'aawide','resnext','densenet','resnet','inception','pyramid','dpn']
	image_dim=(512,512)
	datafiles  = { 'pa':
			{ 'train' : 'small_pa_sample.txt',
		  	  'val'   : 'small_pa_sample.txt'
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

        #can compress below using a "for stage in ['top','all']" loop
		# Train Top (fc) layer
		trainer.set_optimizable_parameters('top')  		# Freeze bottom of model so we are only training the top linear layer
		print("Training Top:", trainer.count_optimizable_parameters(), "Parameters")

		optimizer_top = torch.optim.Adam(trainer.optimizable_parameters, lr = trainer.lr_top)
		lr_scheduler_top = torch.optim.lr_scheduler.StepLR(optimizer_top, step_size=10, gamma=0.5)

		b_f1 = trainer.train('top', criterion, optimizer_top, scheduler = lr_scheduler_top, best_score=0)#model, dataloaders_top, criterion, optimizer_top, trainer.epochs_top, scheduler = lr_scheduler_top, best_acc=0)
		print('Finished training top, best acc {:.4f}'.format(b_f1))

        #now optimize full model
		trainer.set_optimizable_parameters('all')
		print("Training All:", trainer.count_optimizable_parameters(), "Parameters")

		optimizer_all = torch.optim.Adam(trainer.optimizable_parameters, lr = trainer.lr_all)
		lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, step_size=10, gamma=0.8)

		b_f1 = trainer.train('top', criterion, optimizer_all, scheduler = lr_scheduler_all, best_score=b_f1)
		print('Finished training bottom, best acc {:.4f}'.format(b_f1))

		performer=Evalulator.Evaluator(data_type=data_type,modelname=modelname,transform=transform_test)
		print("Finished Performer class on test")
