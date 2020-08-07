import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.distributed as dist
import torchvision
from torchvision import transforms
import PIL
import yaml

import Evaluator
import Trainer
from marsh_plant_dataset import MarshPlant_Dataset_pa, MarshPlant_Dataset_pc


import pdb


if __name__ == "__main__":
	print("PyTorch Version: ",torch.__version__)
	#inputs
	data_type="pa"
	#config_files = ['./config_files/resnext_data1_aug0.yaml', './config_files/resnext_data1_aug1.yaml',
	#				'./config_files/resnext_data2_aug0.yaml', './config_files/resnext_data2_aug1.yaml',
	#				'./config_files/resnext_data3_aug0.yaml', './config_files/resnext_data3_aug1.yaml'
	#				]
	config_files = ['./config_files/config_test.yaml']##['./config_files/resnext_data3_aug0.yaml'] #['./config_files/config_test_3.yaml']#
	#modellist= ['resnext']  #['dpn']#,'neat' 'aawide','resnext','densenet','resnet','inception','pyramid','dpn']
	image_dim=(512,512)
	crop_dim = (1000,1000)

	distributed=False

	for config in config_files:  #convert this to loop on info specified in config file
		try:
			ymlfile = open(config, 'r')
		except IOError:
			continue
		ymldata = yaml.load(ymlfile, Loader=yaml.FullLoader);

		modelname = ymldata["model"]
		datafiles  = { 'pa':
				{ 'train' : ymldata["datafiles"]["train"], #['small_pa_sample.txt'],
			  	  'val'   : ymldata["datafiles"]["val"],
				  'test'  : ymldata["datafiles"]["test"]
			},

			'pc':
		 		{ 'train' : 'marsh_percent_cover_train.txt',
		 		  'val'   : 'marsh_percent_cover_val.txt',
				  'test'   : 'marsh_percent_cover_val.txt'
		 		}

		 }

		train_params = {
			'batch_size_top' : ymldata["batch_size"]["top"],
			'batch_size_all' : ymldata["batch_size"]["all"],
			'epochs_top' : ymldata["epochs"]["top"] ,
			'epochs_all' : ymldata["epochs"]["all"]
		}
		do_data_aug = ymldata["data_aug"]
		#pdb.set_trace()

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
		transforms_base = transforms.Compose([
	    	transforms.Resize(image_dim),
	    	transforms.ToTensor(),
	    	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		if(do_data_aug):
			print("doing data aug")
			transform_train = transforms.Compose([
				transforms.RandomVerticalFlip(),
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(hue=.02, saturation=.02),
				transforms.RandomAffine(20, translate=None, scale = (0.8, 1.1), shear = 10,
					resample = PIL.Image.BILINEAR, fillcolor=0),
					transforms.CenterCrop(crop_dim),
				transforms_base
			])
		else:
			transform_train = transforms_base

		transform_test = transforms_base;
		transform_val = transforms_base;

		#load datasets and setup dataloaders
		if(data_type=="pa"):
			train_data = MarshPlant_Dataset_pa(datafiles[data_type]['train'], train=True ,transform=transform_train)
			val_data   = MarshPlant_Dataset_pa( datafiles[data_type]['val'] , train=True ,transform=transform_val)
			test_data  = MarshPlant_Dataset_pa( datafiles[data_type]['test'], train=True ,transform=transform_test)
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

		results = {'best_score': 0.0, 'best_model': []}
		gamma = {'top' :0.5, 'all': 0.8}
		for stage in ['top', 'all']:
		# Train Top (fc) layer
			trainer.set_optimizable_parameters(stage)
			print("Training {}: {} parameters optimized".format(stage, trainer.count_optimizable_parameters() ) )

			optimizer = torch.optim.Adam(trainer.optimizable_parameters, lr = trainer.lr[stage])
			lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma[stage])

			results = trainer.train(stage, criterion, optimizer, scheduler = lr_scheduler, results = results)
			print('Finished training {}, best acc {:.4f}'.format(stage, results['best_score']))

		performer=Evaluator.Evaluator(results['best_model'],data_type=data_type,modelname=modelname,transform=transform_test, config_file = config)
		performer.setup_dataloader(test_data)
		performer.run()
		print("Finished Performer class on test")
