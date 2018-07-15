from __future__ import print_function

import os
import time
import datetime
import random
import json
import argparse
import densenet
import numpy as np
import keras.backend as K

from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import np_utils

from keras.layers.core import Dense
from keras.models import Model
from keras.layers import Input

import data_loader

def run_MURA(batch_size,
				nb_epoch,
				depth,
				nb_dense_block,
				nb_filter,
				growth_rate,
				dropout_rate,
				learning_rate,
				weight_decay,
				plot_architecture):
	""" Run MURA experiments

	:param batch_size: int -- batch size
	:param nb_epoch: int -- number of training epochs
	:param depth: int -- network depth
	:param nb_dense_block: int -- number of dense blocks
	:param nb_filter: int -- initial number of conv filter
	:param growth_rate: int -- number of new filters added by conv layers
	:param dropout_rate: float -- dropout rate
	:param learning_rate: float -- learning rate
	:param weight_decay: float -- weight decay
	:param plot_architecture: bool -- whether to plot network architecture

	"""

	###################
	# Data processing #
	###################

	
	im_size = 128 #测试修改参数 size root_path nb_epoch nb_dense_block  model.save   #/home/tang/datasets/MURA-v1.1/train/XR_ELBOW  XR_SHOULDER
	train_path = '../valid/XR_FOREARM'
	valid_path = '../valid/XR_FOREARM'
	X_train_path, Y_train = data_loader.load_path(root_path = train_path,size = im_size)   # XR_FOREARM  XR_FINGER  XR_HAND  XR_HUMERUS  XR_WRIST 
	X_valid_path, Y_valid = data_loader.load_path(root_path = valid_path, size = im_size)  # root_path = '../valid/XR_ELBOW'

	X_valid = data_loader.load_image(X_valid_path,im_size)  #提前加载验证集？
	Y_valid = np.asarray(Y_valid)
	# nb_classes = len(np.unique(Y_train))
	nb_classes = 1
	img_dim = (im_size,im_size,1)#X_train.shape[1:]+(1,) #加上最后一个维度,类型为tuple

	'''
注释掉不用的代码
	# if K.image_data_format() == "channels_first":
	#     n_channels = X_train.shape[1]
	# else:
	#     n_channels = X_train.shape[-1]


	X_train = X_train.astype('float32')
	X_valid = X_valid.astype('float32')

	# Normalisation
	X = np.vstack((X_train, X_valid))
	# 2 cases depending on the image ordering
	if K.image_data_format() == "channels_first":
		pass #只用tensorlflow
		# for i in range(n_channels):
		#     mean = np.mean(X[:, i, :, :])
		#     std = np.std(X[:, i, :, :])
		#     X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
		#     X_valid[:, i, :, :] = (X_valid[:, i, :, :] - mean) / std

	elif K.image_data_format() == "channels_last":
			mean = np.mean(X[:, :, :])
			std = np.std(X[:, :, :])
			X_train[:, :, :] = (X_train[:, :, :] - mean) / std
			X_valid[:, :, :] = (X_valid[:, :, :] - mean) / std
			X_valid = np.expand_dims(X_valid,axis=3)         #扩展维度3
			# print(X_valid.shape)
		# for i in range(n_channels):
		#     mean = np.mean(X[:, :, :, i])
		#     std = np.std(X[:, :, :, i])
		#     X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
		#     X_valid[:, :, :, i] = (X_valid[:, :, :, i] - mean) / std
	'''


	###################
	# Construct model #
	###################

	model = densenet.DenseNet(nb_classes,
							  img_dim,
							  depth,
							  nb_dense_block,
							  growth_rate,
							  nb_filter,
							  dropout_rate=dropout_rate,
							  weight_decay=weight_decay)
	# Model output
	# model.summary()

	# Build optimizer
	opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


	#load pretrained model 加载以前训练好的模型
	model_name = '1test'
	pretrained_model_path = 'save_models/MURA_model_'+model_name+'.h5'
	if os.path.exists(pretrained_model_path):   
		print('load pretrained model...')
		model = load_model(pretrained_model_path, compile=False)        #base_model
		# x = model.get_layer('global_average_pooling2d_1').output
		# x = Dense(256, activation='relu',)(x)
		# x = Dense(1,activation='sigmoid',)(x)
		# model = Model(inputs = model.input,outputs = x) 
		# for layer in model.layers[:-1]:			#在训练好的模型后面，添加dense层之后， 微调fine-tune 最后一层
		# 	layer.trainable = False
		model.compile(loss='binary_crossentropy',
				  optimizer=opt,
				  metrics=["accuracy"])
	else:
		print('no pretrained model')
		model.compile(loss='binary_crossentropy',
					  optimizer=opt,
					  metrics=["accuracy"])
	model.summary()


	if plot_architecture:
		from keras.utils import plot_model
		plot_model(model, to_file='./figures/densenet_archi.png', show_shapes=True)

	####################
	# Network training #
	####################

	print("Start Training")

	list_train_loss = []
	list_valid_loss = []
	list_train_loss_all = []
	list_learning_rate = []
	best_record = [100,0,100,100] #记录最优 [验证集损失函数值,准确率，训练集数据集loss差值,acc差值]
	start_time = datetime.datetime.now()
	for e in range(nb_epoch):

		# if e == 1:					#在训练好的模型后面，添加dense层之后， 微调fine-tune 最后一层
		# 	if os.path.exists(pretrained_model_path):   
		# 		print('set trainable')
		# 		for layer in model.layers[:-1]:
		# 			layer.trainable = True
		# 		model.compile(loss='binary_crossentropy',
		# 				  optimizer=opt,
		# 				  metrics=["accuracy"])
		# 		model.summary()
		if e == int(0.25 * nb_epoch):
			K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

		if e == int(0.5 * nb_epoch):
			K.set_value(model.optimizer.lr, np.float32(learning_rate / 50.))

		if e == int(0.75 * nb_epoch):
			K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

		X_train_path, Y_train = data_loader.load_path(root_path = valid_path,size = im_size)   ## root_path = '../train
		X_valid_path, Y_valid = data_loader.load_path(root_path = valid_path, size = im_size)  # root_path = '../valid/XR_ELBOW'
		X_valid = data_loader.load_image(X_valid_path,im_size)  #提前加载验证集？
		Y_valid = np.asarray(Y_valid)


		split_size = batch_size
		num_splits = len(X_train_path) / split_size
		arr_all = np.arange(len(X_train_path)).astype(int)
		random.shuffle(arr_all)                 #随机打乱index索引顺序
		arr_splits = np.array_split(arr_all, num_splits)

		l_train_loss = []
		batch_train_loss = []
		start = datetime.datetime.now()

		for i,batch_idx in enumerate(arr_splits):


			X_batch_path,Y_batch = [],[]
			for idx in batch_idx:
				X_batch_path.append(X_train_path[idx])
				Y_batch.append(Y_train[idx])
			X_batch = data_loader.load_image(Path = X_batch_path, size =im_size)
			Y_batch = np.asarray(Y_batch)
			train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)

			l_train_loss.append([train_logloss, train_acc])
			batch_train_loss.append([train_logloss, train_acc])
			if i %100 == 0:
				loss_1, acc_1 = np.mean(np.array(l_train_loss), 0)
				loss_2, acc_2 = np.mean(np.array(batch_train_loss), 0)
				batch_train_loss = []           #当前100batch的损失函数和准确率
				print ('[Epoch {}/{}] [Batch {}/{}] [Time: {}] [all_batchs--> train_epoch_logloss: {:.5f}, train_epoch_acc:{:.5f}] '.format
					(e+1,nb_epoch,i, len(arr_splits),datetime.datetime.now() - start,loss_1,acc_1),
					'[this_100_batchs-->train_batchs_logloss: {:.5f}, train_batchs_acc:{:.5f}]'.format(loss_2, acc_2))



		# X_valid = data_loader.load_image(X_valid_path,im_size)
		# Y_valid = np.asarray(Y_valid)
		valid_logloss, valid_acc = model.evaluate(X_valid,
												Y_valid,
												verbose=0,
												batch_size=64)
		list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
		list_valid_loss.append([valid_logloss, valid_acc])
		list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
		# to convert numpy array to json serializable
		print('[Epoch %s/%s] [Time: %s, Total_time: %s]' % (e + 1, nb_epoch, datetime.datetime.now() - start,
			datetime.datetime.now() - start_time),end = '')
		print('[train_loss_and_acc:{:.5f} {:.5f}] [valid_loss_acc:{:.5f} {:.5f}]'.format(list_train_loss[-1][0],
			list_train_loss[-1][1],list_valid_loss[-1][0],list_valid_loss[-1][1]))

		# list_train_loss_all.append(np.array(l_train_loss).tolist())
		list_train_loss_all+=np.array(l_train_loss).tolist()

		d_log = {}
		d_log["batch_size"] = batch_size
		d_log["nb_epoch"] = nb_epoch
		d_log["optimizer"] = opt.get_config()
		d_log["train_loss"] = list_train_loss
		d_log["valid_loss"] = list_valid_loss
		d_log["train_loss_all"] = list_train_loss_all
		d_log["learning_rate"] = list_learning_rate

		json_file = os.path.join('./log/experiment_log_MURA_%s.json'%(model_name))
		with open(json_file, 'w') as fp:
			json.dump(d_log, fp, indent=4, sort_keys=True)

		record = [valid_logloss,valid_acc,abs(valid_logloss-list_train_loss[-1][0]),abs(valid_acc-list_train_loss[-1][1]),]
		if ((record[0]<=best_record[0]) &(record[1]>=best_record[1])) :
			if e <= int(0.25 * nb_epoch)|(record[2]<=best_record[2])&(record[3]<=best_record[3]):#四分之一epoch之后加入差值判定
				best_record=record                      #记录最小的 [验证集损失函数值,准确率，训练集数据loss差值,acc差值]
				print('saving the best model:epoch',e+1,best_record)
				# model.save('save_models/best_MURA_model.h5')
				# model.save('save_models/best_MURA_model@epochs{}.h5'.format(e+1))
		model.save('save_models/MURA_model_{}@epochs{}.h5'.format(model_name,e+1))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Run MURA experiment')
	parser.add_argument('--batch_size', default=8, type=int, #default=64
						help='Batch size')
	parser.add_argument('--nb_epoch',  type=int, default=10,#default=30,
						help='Number of epochs')
	parser.add_argument('--depth', type=int, default=6*3+4,#default=7,
						help='Network depth')
	parser.add_argument('--nb_dense_block', type=int, default=1, #default=1,
						help='Number of dense blocks')
	parser.add_argument('--nb_filter', type=int, default=16,
						help='Initial number of conv filters')
	parser.add_argument('--growth_rate', type=int, default=12,
						help='Number of new filters added by conv layers')
	parser.add_argument('--dropout_rate', type=float, default=0.2,
						help='Dropout rate')
	parser.add_argument('--learning_rate', type=float, default=1E-4, #default=1E-3,
						help='Learning rate')
	parser.add_argument('--weight_decay', type=float, default=1E-4,
						help='L2 regularization on weights')
	parser.add_argument('--plot_architecture', type=bool, default=False,#default=False,
						help='Save a plot of the network architecture')

	args = parser.parse_args()

	print("Network configuration:")
	for name, value in parser.parse_args()._get_kwargs():
		print(name, value)

	list_dir = ["./log", "./figures", "./save_models"]
	for d in list_dir:
		if not os.path.exists(d):
			os.makedirs(d)

	run_MURA(args.batch_size,
				args.nb_epoch,
				args.depth,
				args.nb_dense_block,
				args.nb_filter,
				args.growth_rate,
				args.dropout_rate,
				args.learning_rate,
				args.weight_decay,
				args.plot_architecture)
