import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from modeling.backbone.resnet import ResNet101
from RN101_newtop import RN101_newtop
from PerformanceMetrics import PerformanceMetrics

class Trainer(object):
	def __init__(self, train_params, data_type="pa", modelname="resnet"):
		self.batch_size = {}
		self.epochs = {}
		self.lr = {}
		self.dataloaders = { 'top' : {}, 'all' :{} }
		self.model = []
		self.criterion = []
		self.optimizable_parameters = []

		self.batch_size['top'] = train_params['batch_size_top']
		self.batch_size['all'] = train_params['batch_size_all']
		self.epochs['top'] = train_params['epochs_top'] #30 number of epochs to train the top of the model
		self.epochs['all'] = train_params['epochs_all'] #20 number of epochs to train the entire model
		self.lr['top'] = 1e-4
		self.lr['all'] = 1e-5

		self.data_type=data_type
		self.modelname=modelname
		if(data_type=="pa"):
			self.N_CLASSES = 7
		elif(data_type=='pc'):
			self.N_CLASSES = 9

		self.log_file = open("marsh_plant_nn_training_logfile.txt","w")
		self.model_path = './modeling/saved_models/'+modelname+'_'+data_type+'.torch'



###
# DataLoader Setup
###
	def setup_dataloaders(self,dataset_dict, bShuffle,num_workers,samplers):
		for stage in self.batch_size:
			for phase in dataset_dict:
				if samplers is not None:
					self.dataloaders[stage][phase] =  torch.utils.data.DataLoader(dataset_dict[phase], batch_size = self.batch_size[stage], shuffle = bShuffle, num_workers = num_workers,sampler=samplers[phase])
				else:
					self.dataloaders[stage][phase] =  torch.utils.data.DataLoader(dataset_dict[phase], batch_size =  self.batch_size[stage], shuffle = bShuffle, num_workers = num_workers)


###
# Model Setup
###

	def setup_model(self, distributed):
		if(self.modelname=="resnet"):
			pretrained_model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
			resnet_bottom = torch.nn.Sequential(*list(pretrained_model.children())[:-1]) # remove last layer (fc) layer
			model = RN101_newtop(base_model = resnet_bottom, num_classes = self.N_CLASSES)
		elif(self.modelname=='densenet'):
			model = models.densenet121(pretrained=True)
			num_ftrs = model.classifier.in_features
			model.classifier = nn.Linear(num_ftrs,self.N_CLASSES)
		elif(self.modelname=='inception'):
			model= models.inception_v3(pretrained=True)
			#set_parameter_requires_grad(model_ft, feature_extract)
			# Handle the auxilary net
			num_ftrs = model.AuxLogits.fc.in_features
			model.AuxLogits.fc = nn.Linear(num_ftrs, self.N_CLASSES)
			# Handle the primary net
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs,self.N_CLASSES)
		elif(self.modelname=='resnext'):
			self.batch_size_all=4 # got cuda error
			model= models.resnext101_32x8d(pretrained=True)
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs,self.N_CLASSES)
		elif(self.modelname=='pyramid'):
			model= prn.PyramidNet(dataset='imagenet', depth=101, alpha=360, num_classes=1000, bottleneck=True,pretrain=True) #input imagenet args
			num_ftrs = model.fc.in_features
			#model = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
			model.fc = nn.Linear(num_ftrs,self.N_CLASSES)
		elif(self.modelname=='dpn'):
			model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn92', pretrained=True)
			num_chs = model.classifier.in_channels#*self.batch_size_all
			model.classifier = nn.Conv2d(num_chs, self.N_CLASSES, kernel_size=1, bias=True)
		elif(self.modelname=='aawide'):
			model = Wide_ResNet(depth=10, widen_factor=5, dropout_rate=0.3, num_classes=self.N_CLASSES, shape=32)
			self.lr_all= 1e-3
			self.epochs_all=100
		elif(self.modelname=='neat'):
			model = NeatCNN(num_classes=self.N_CLASSES,channel_size=256,group_size=2,depth=1,width=1,residual=False)
			self.epochs_all=50
			self.batch_size_all=4
		elif(self.modelname=='neater'):
			model = NeatCNN(num_classes=self.N_CLASSES,channel_size=256,group_size=2,depth=3,width=1,residual=True)
			self.epochs_all=50
			self.batch_size_all=4

		print(model)
		model.cuda()
		if distributed:
			model = torch.nn.parallel.DistributedDataParallel(model)

		self.model =  model

# helper functions
	def count_optimizable_parameters(self):
		return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

	def set_optimizable_parameters(self, stage):
		if stage == 'top':
			for param in self.model.parameters():
				param.requires_grad = False
			if(self.modelname=='densenet' or self.modelname=='dpn'or self.modelname=='neat' or  self.modelname=='neater'):
				params_to_optimize_in_top = list(self.model.classifier.parameters())
			else:
				params_to_optimize_in_top = list(self.model.fc.parameters())

			for param in params_to_optimize_in_top:
					param.requires_grad = True

			self.optimizable_parameters = params_to_optimize_in_top

		if stage == 'all':
			for param in self.model.parameters():
				param.requires_grad = True

			self.optimizable_parameters = self.model.parameters()

	def evaluate_batch(self, input, target, phase):
		with torch.set_grad_enabled(phase =='train'):  #track gradients for backprop in training phase
			sigfunc = nn.Sigmoid()
			if(self.modelname=='inception' and phase=='train'):
				output, aux_output = self.model(input)

				##calculate loss - move this to a function: calculate_loss(output, target)
				if(self.data_type=='pc'):
					target=target.long()
					#print(target)
					label=torch.max(target, 1)[1]
					loss1 = self.criterion(output, label)
					loss2 = self.criterion(aux_output, label)
					loss = loss1 + 0.4*loss2
				else:
					loss1 = self.criterion(output, target)
					loss2 = self.criterion(aux_output, target)
					loss = loss1 + 0.4*loss2  # evaluate loss

			else :
				output = self.model(input)  # pass in image series
				if(self.data_type=='pc'):
					target=target.long()
					#print(target)
					label=torch.max(target, 1)[1]
					loss = self.criterion(output, label)  # evaluate loss
				else:
					loss = self.criterion(output, target)  # evaluate loss


			#make predictions
			if(self.data_type == 'pc'):
				print("check how to make predictions for pc")
			else:
				sig = sigfunc(output)
				sig = sig.to("cpu").detach().numpy()
				pred = sig > 0.5 #make this threshold variable
				pred = pred.astype(int)

		return pred, loss


	def train(self, stage, criterion, optimizer, scheduler = None,  best_score=0 ):
		self.criterion = criterion
		for epoch in range(self.epochs[stage]):
			for phase in ['train','val']:
				if phase == 'train':
					self.model.train()
				else:
					self.model.eval()

				metrics = PerformanceMetrics()

				for it, batch in enumerate(self.dataloaders[stage][phase]):
					input = batch['X'].cuda()#to(device)
					target = batch['Y'].cuda()#to(device)

					optimizer.zero_grad()  #zero gradients

					pred, loss = self.evaluate_batch(input, target, phase)
					if phase == 'train':
						loss.backward()  # update the gradients
						optimizer.step()  # update sgd optimizer lr

					# compare model output and annotations  - more easily done with numpy
					target = target.to("cpu").detach().numpy()  #take off gpu, detach from gradients
					target = target.astype(int)

					n_samples =  input.size(0)
					metrics.accumulate(loss.item(), n_samples, pred, target)

					if it % 50 == 0:
						print('Epoch: {}, Iteration: {}, Loss: {:.4f}, F1_score: {:.4f}'.format(epoch, it, metrics.loss_per_sample, metrics.f1))

				if scheduler:
					scheduler.step()
					print("Learning rate :")
					print(scheduler.get_lr())

				# save model if it is the best model on val set
				print('{} Loss: {:.4f} F1_score: {:.4f}'.format(phase, metrics.loss_per_sample, metrics.f1))
				self.log_file.write('epoch\t{}\tphase\t{}\tLoss\t{:.4f}\tPrecision\t{:.4f}\n'.format(epoch, phase,  metrics.loss_per_sample, metrics.f1))

				if phase == 'val' and metrics.f1 > best_score:
					best_score = metrics.f1
					torch.save(self.model, self.model_path)# save state dict, this method is bound to break.

		return best_score
