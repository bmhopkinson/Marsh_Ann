import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from modeling.backbone.resnet import ResNet101
from marsh_plant_dataset import MarshPlant_Dataset_pa, MarshPlant_Dataset_pc
from RN101_newtop import RN101_newtop
from marsh_plant_nn_performance import Performer
from Trainer import Trainer
#import PyramidNet_model.PyramidNet as prn
import torch.utils.data.distributed
import torch.distributed as dist
#from Attention_Augmented.attention_augmented_wide_resnet import Wide_ResNet
#from neatcnn import NeatCNN

#for batch in data_loader:
#    output = model(batch['X'].to(device))
#    loss = criterion(output, batch['Y'].to(device))
#    #print(batch['Y'])
#    print(loss)


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
		'epochs_top' : 20 ,
		'epochs_all' : 20
	 }

	distributed=False


	for modelname in modellist:

		trainer = Trainer(datafiles, train_params, data_type=data_type,modelname=modelname)
		if(modelname=="pyramid" ): #or modelname=='dpn'):
			image_dim=(224,224)
		if(modelname=="aawide" ):
			image_dim=(32,32)
		#if(modelname=="dpn"):
			#distributed=True
		if distributed :
			dist.init_process_group(backend='gloo', init_method='env://',
                                rank=0,world_size=4)

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



		if(data_type=="pa"):
			train_data = MarshPlant_Dataset_pa(trainer.train_infile,transform=transform_train)
			val_data  = MarshPlant_Dataset_pa(trainer.val_infile,transform=transform_val)
			datasets = {'train' : train_data, 'val' : val_data}
			criterion = nn.BCEWithLogitsLoss().cuda()
			if distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
				val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
				samplers={'train' : train_sampler, 'val' : val_sampler}
		if(data_type=="pc"):
			train_data = MarshPlant_Dataset_pc(trainer.train_infile,transform=transform_train)
			val_data  = MarshPlant_Dataset_pc(trainer.val_infile,transform=transform_val)
			datasets = {'train' : train_data, 'val' : val_data}
			criterion = nn.CrossEntropyLoss().cuda()#change to cross entropy here.
			if distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
				val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
				samplers={'train' : train_sampler, 'val' : val_sampler}

		bShuffle = True
		num_workers = 8
		if not distributed:
			samplers=None
		dataloaders_top = trainer.setup_dataloader(datasets, trainer.batch_size_top, bShuffle, num_workers,samplers=samplers)
		dataloaders_all = trainer.setup_dataloader(datasets, trainer.batch_size_all, bShuffle, num_workers,samplers=samplers)

		#setup model and training parameters

		model = trainer.setup_model()
		print(model)
		model.cuda()
		if distributed:
			model = torch.nn.parallel.DistributedDataParallel(model)

		# Train Top (fc) layer
		# Freeze bottom of model so we are only training the top linear layer
		for param in model.parameters():
			param.requires_grad = False
		if(modelname=='densenet' or modelname=='dpn'or modelname=='neat' or  modelname=='neater'):
			params_to_optimize_in_top = list(model.classifier.parameters())
		else:
			params_to_optimize_in_top = list(model.fc.parameters())

		for param in params_to_optimize_in_top:
			param.requires_grad = True

		optimizer_top = torch.optim.Adam(params_to_optimize_in_top, lr = trainer.lr_top)
		lr_scheduler_top = torch.optim.lr_scheduler.StepLR(optimizer_top, step_size=10, gamma=0.5)
		#lr_scheduler_top = None
		print("Training Top:", trainer.count_parameters(model), "Parameters")

		b_acc = trainer.train(model, dataloaders_top, criterion, optimizer_top, trainer.epochs_top, scheduler = lr_scheduler_top, best_acc=0)
		print('Finished training top, best acc {:.4f}'.format(b_acc))


		for param in model.parameters():
			param.requires_grad = True
		print("Training All:", trainer.count_parameters(model), "Parameters")

		# Optimizer for Entire Network
		optimizer_all = torch.optim.Adam(model.parameters(), lr = trainer.lr_all)
		lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, step_size=10, gamma=0.8)
		#lr_scheduler_all = None
		# Train on All

		b_acc = trainer.train(model, dataloaders_all, criterion, optimizer_all, trainer.epochs_all, scheduler = lr_scheduler_all, best_acc=b_acc)
		print('Finished training bottom, best acc {:.4f}'.format(b_acc))

		performer=Performer(data_type=data_type,modelname=modelname,transform=transform_test)
		print("Finished Performer class on test")
