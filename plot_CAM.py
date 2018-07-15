# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import cv2
import os
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import data_loader

def plot_CAM():

	im_size = 320
	path_CAM = '../valid/XR_HAND'  #XR_SHOULDER'     #XR_WRIST    #  /XR_ELBOW'
	# Path = []
	# for f_path in os.listdir('test_images'):
	# 	Path.append(os.path.join('test_images',f_path))
	# 	
	Path,labels = data_loader.load_path(root_path = path_CAM ,size = im_size)
	[train_or_valid,studytype,patient_ID,study,im_name,label] = data_loader.path_detail(Path)	
	#load pretrained model 加载以前训练好的模型
	pretrained_model_path = 'save_models/MURA_model_XR_ELBOW@epochs44.h5' 
	pretrained_model_path = 'save_models/best_MURA_model.h5'
	pretrained_model_path = 'save_models/'+'MURA_model_dense170@epochs5.h5' #'save_models/best_MURA_modle.h5'
	
	# if os.path.exists(pretrained_model_path):   
	print('load pretrained model from '+pretrained_model_path)
	model = load_model(pretrained_model_path)		#  , compile=False
	# model.summary()

	in_layer = model.layers[0].input
	out_layer =  model.layers[-1].output
	last_conv = model.layers[-3].output
	Weights = model.layers[-1].get_weights()[0]		#返回是权重和偏置array的列表，只取权重

	model = Model(inputs = [in_layer], outputs = [last_conv,out_layer])
	opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss='binary_crossentropy',
	  optimizer=opt,
	  metrics=["accuracy"])
	model.summary()

	for i,path in enumerate(Path):

		img = data_loader.load_image([path], size = im_size, rotation=False, normalization=True)
		# get_output = K.function([in_layer], [last_conv,out_layer])
		# [conv_outputs, predictions_1] = get_output([img])
		[conv_outputs, predictions_1] = model.predict(img)
		CAM = np.dot(conv_outputs,Weights)
		predictions = int(np.around(predictions_1))			#predictions_1.shape =array([[0.345]])

		CAM_im = CAM[0, :, :, :]  #只选array的后三维
		CAM_im = cv2.resize(CAM_im, (im_size, im_size)) #还原成原来的大小

		img_ori = data_loader.load_image([path], size = im_size, rotation=False, normalization=False)
		img_ori = np.squeeze(img_ori)		# img[0, :, :, 0]  #只要两维的灰度图
		# print(CAM.shape)
		# print(img_ori.shape)
		# print(Weights.shape)
		# print(last_conv.shape)

		normal_abnomarl = ['negative','positive']
		# imshow(X,...)其中，X变量存储图像，可以是浮点型数组、unit8数组以及PIL图像，如果其为数组，则需满足一下形状：
		#    (1) M*N      此时数组必须为浮点型，其中值为该坐标的灰度；
		#    (2) M*N*3  RGB（浮点型或者unit8类型）
		#    (3) M*N*4  RGBA（浮点型或者unit8类型）
		#    介绍网址：  https://blog.csdn.net/Eastmount/article/details/73392106?locationNum=5&fps=1
		
		# #原始图片和CAM分别作图
		# fig = plt.figure(normal_abnomarl[predictions]+path)
		# ax1 = fig.add_subplot(121)
		# im1 = ax1.imshow(img_ori)
		# plt.colorbar(im1, shrink=0.5)
	 
		# ax2 = fig.add_subplot(122)
		# im2 = ax2.imshow(CAM_im,
		#            cmap="jet",
		#            alpha=0.5,
		#            interpolation='nearest')
		# plt.colorbar(im2, shrink=0.5)
		# # plt.show()
		# # fig.savefig('./figures/plot_CAM%s.jpg'%(normal_abnomarl[predictions]+os.path.split(path)[-1]))

		#原始图片和CAM一起作图
		fig1 = plt.figure(normal_abnomarl[predictions])
		plt.imshow(img_ori)
		plt.imshow(CAM_im,
		           cmap="jet",
		           alpha=0.35,
		           interpolation='nearest')
		# plt.show()
		# fig1.savefig('./figures/plot_CAM_1%s.jpg'%(normal_abnomarl[predictions]+os.path.split(path)[-1]))

		
		# 创建保存目录	#true_negative_n:  predictions=negative   	labels=negative    True
		save_dirs= ['true_negative_n','false_negative_p','false_positive_n','true_positive_p']  
		for d in save_dirs:
			d = os.path.join('figures',d)
			if not os.path.exists(d):
				os.makedirs(d)
		print(predictions,labels[i],predictions*2+labels[i],save_dirs[predictions*2+labels[i]])

		fig1.savefig('./figures/%s/%s.jpg'%(save_dirs[predictions*2+labels[i]],
			studytype[i]+str(round(float(predictions_1),3)) +patient_ID[i]+im_name[i] ))
		plt.clf()
		plt.close('all')


	return 


def get_accuracy():				
	#MURA_model_dense170@epochs25.h5  模型验证集准确率  0.69557964970809
	path_load = '../valid_labeled_studies.csv'# '../valid_labeled_studies.csv' valid_labeled_studies_output_dense170						#按照study计算准确率
	path_save = '../valid_labeled_studies_output.csv'
	path_dataset = '../../'
	im_size = 320  # 320  #128 for test

	dataframe = pd.read_csv(path_load,header=None,index_col=None)
	Path = list(dataframe[0])			#取部分验证数据 XR_ELBOW:[866:1024]	XR_HAND:[370:537]  XR_HUMERUS:[537:672]
	Label = list(dataframe[1])	  		# XR_SHOULDER: [672:866]  XR_FINGER:[1024:1199]  XR_WRIST:[0:237]  XR_FOREARM:[237:370]
	
	# outputs_l = list(dataframe[2])
	# acc_out = list(dataframe[3])
	print(len(Label))

	# 'load pretrained model '
	pretrained_model_path = 'save_models/'+'MURA_model_dense170@epochs44.h5'
	pretrained_model_path = 'save_models/'+'MURA_model_dense170@epochs25.h5' #'MURA_model_XR_ELBOW@epochs44.h5'	#加载模型
	pretrained_model_path = 'save_models/'+'MURA_model_dense169@epochs23.h5'
	# pretrained_model_path = 'save_models/'+'best_MURA_model.h5'		#for test

	print('load pretrained model from '+pretrained_model_path)
	model = load_model(pretrained_model_path)		
	model.summary()

	outputs = [0]*len(Path)
	for idx,path in enumerate(Path):			
		#计算每个study下图片的预测值
		path = path_dataset+path
		im_paths = os.listdir(path)
		im_paths = [os.path.join(path,im_path) for im_path in im_paths]

		im = data_loader.load_image(Path=im_paths, size=im_size, rotation=False)   #不加入随机旋转
		predictions = model.predict(im)
		outputs[idx] = float(np.mean(predictions))
	outputs_l = [round(outputs[i]) for i in range(len(outputs))]			#输出四舍五入

	# 计算准确率
	acc_out = [1-(Label[i]^outputs_l[i]) for i in range(len(outputs_l))]  	#1-异或   相同为1，不同为0
	accuracy = np.mean(acc_out)
	print(accuracy)

	# 每一类准确率
	[train_or_valid,studytype,patient_ID,study,im_name,label] = data_loader.path_detail(Path=Path)
	type_name = list(set(studytype))		# ['XR_WRIST', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_ELBOW','XR_FINGER']
	type_start = {name : studytype.index(name)  for name in type_name}
	type_count = {name : studytype.count(name) for name in type_name}
	type_acc =   {name : np.mean(acc_out[type_start[name]:type_start[name]+type_count[name]]) for name in type_name}
	print(type_acc,np.mean(acc_out))

	print(type_count)

	#保存到csv文件
	save_output = [[Path[i], Label[i], outputs_l[i],acc_out[i]] for i in range(len(outputs_l))]
	df_save = pd.DataFrame(save_output)
	df_save.to_csv(path_save,header=None,index=None)

	return

def csv_acc(path_label = '../valid_labeled_studies.csv', path_output ='../valid_labeled_studies_output.csv'):
	# 计算输出结果csv与标注的准确率
	dataframe_1 = pd.read_csv(path_label,header=None,index_col=None)
	Path_1 = list(dataframe_1[0])
	Label = list(dataframe_1[1])	  		
	dataframe_2 = pd.read_csv(path_output,header=None,index_col=None)
	Path_2 = list(dataframe_2[0])
	outputs = list(dataframe_2[1])

	# 计算准确率
	acc_out = []
	for i ,path in enumerate(Path_1):
		idx = Path_2.index(path)
		predict = 1-(Label[i]^outputs[idx]) 
		acc_out.append(predict)
	# acc_out = [1-(Label[i]^outputs_l[i]) for i in range(len(outputs_l))]  	#1-异或   相同为1，不同为0
	accuracy = np.mean(acc_out)
	print(accuracy)


	[train_or_valid,studytype,patient_ID,study,im_name,label] = data_loader.path_detail(Path=Path_1)
	type_name = list(set(studytype))		# ['XR_WRIST', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_ELBOW','XR_FINGER']
	type_start = {name : studytype.index(name)  for name in type_name}
	type_count = {name : studytype.count(name) for name in type_name}
	type_acc =   {name : np.mean(acc_out[type_start[name]:type_start[name]+type_count[name]]) for name in type_name}
	print(type_acc,np.mean(acc_out))

	print(type_count)

	return 


def get_cifar(path = 'test_images/image0_n.png', size = 32):		# 得到图中最值位置的32*32切割图	每次操作单张图片，减少内存

	image = data_loader.load_image([path], size = 320, rotation=False, normalization=False)
	image = np.squeeze(image)
	index = np.where(image == np.max(image))	#返回最大值判断值为真的地址，tuple:(维度0的索引array,维度1的索引array,...)
	x, y = index[0][0], index[1][0]				#第一维度的值，第二维度的值

	size_h = int(size/2)
	if x-size_h<0 | x+size_h>size | y-size_h<0 | y+size_h>size:		#超出图片大小，返回？
		return 
	
	image_clip = image[(x-size_h):(x+size_h),(y-size_h):(y+size_h)]
	# print(x,y,np.max(image),image.shape,image_clip)

	cv2.imshow(path,image_clip)
	# cv2.imwrite('img.png',image_clip)
	# cv2.imwrite('img_o.png',image)
	cv2.waitKey()
	cv2.imshow(path,image)
	cv2.waitKey()

	return

if __name__ == "__main__":
	# plot_CAM()
	# get_accuracy()
	csv_acc()
	get_cifar()