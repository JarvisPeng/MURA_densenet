# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# https://worksheets.codalab.org/worksheets/0x42dda565716a4ee08d61f0a23656d8c0/
# delete this line doker image :pingchesu/keras2-1-3   ashspencil/ml-cuda-app-keras2.1.5   gw000/keras-full
# cl run valid_image_paths.csv:valid_image_paths.csv MURA-v1.1:valid src:src "python src/evaluate_s.py valid_image_paths.csv predictions.csv" -n run-predictions --request-docker-image gw000/keras-full --request-memory 3g
# cl make run-predictions/predictions.csv -n predictions-Densenet169-single-model
# cl macro mura-utils/valid-eval-v1.1 predictions-Densenet169-single-model
# cl edit predictions-Densenet169-single-model --tags mura-submit
def evaluate_accuracy(path_load = '../valid_labeled_studies.csv', path_save = '../valid_labeled_studies_output.csv'):				
	#MURA_model_dense170@epochs25.h5  模型验证集准确率  0.69557964970809
	# path_load = '../valid_labeled_studies.csv'# '../valid_labeled_studies.csv' valid_labeled_studies_output_dense170						#按照study计算准确率
	# path_save = '../valid_labeled_studies_output.csv'
	print(path_load, path_save)
	path_dataset = 'F:/BaiduNetdiskDownload/'			# '../../'
	im_size = 320  # 320  #128 for test

	dataframe = pd.read_csv(path_load,header=None,index_col=None)
	Path = list(dataframe[0])			#取部分验证数据 XR_ELBOW:[866:1024]	XR_HAND:[370:537]  XR_HUMERUS:[537:672]
	Label = list(dataframe[1])	  		# XR_SHOULDER: [672:866]  XR_FINGER:[1024:1199]  XR_WRIST:[0:237]  XR_FOREARM:[237:370]
	Path = [path_dataset+path for path in Path]
	
	# outputs_l = list(dataframe[2])
	# acc_out = list(dataframe[3])
	print(len(Label))

	# 'load pretrained model '
	pretrained_model_path = 'save_models/'+'MURA_model_dense170@epochs44.h5'
	pretrained_model_path = 'save_models/'+'MURA_model_dense170@epochs25.h5' #'MURA_model_XR_ELBOW@epochs44.h5'	#加载模型
	pretrained_model_path = 'save_models/'+'MURA_model_dense169@epochs23.h5'
	# pretrained_model_path = 'save_models/'+'MURA_model_1test.h5'		#for test

	print('load pretrained model from '+pretrained_model_path)
	model = load_model(pretrained_model_path)		
	model.summary()

	outputs = [0]*len(Path)
	for idx,path in enumerate(Path):			
		#计算每个study下图片的预测值
		im_paths = os.listdir(path)
		im_paths = [os.path.join(path,im_path) for im_path in im_paths]

		im = load_image(Path=im_paths, size=im_size)   
		predictions = model.predict(im)
		outputs[idx] = float(np.mean(predictions))
	outputs_l = [round(outputs[i]) for i in range(len(outputs))]			#输出四舍五入

	# 计算准确率
	acc_out = [1-(Label[i]^outputs_l[i]) for i in range(len(outputs_l))]  	#1-异或   相同为1，不同为0
	accuracy = np.mean(acc_out)
	print(accuracy)

	[train_or_valid,studytype,patient_ID,study,im_name,label] = path_detail(Path=Path)
	type_name = list(set(studytype))		# ['XR_WRIST', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_ELBOW','XR_FINGER']
	type_start = {name : studytype.index(name)  for name in type_name}
	type_count = {name : studytype.count(name) for name in type_name}
	type_acc =   {name : np.mean(acc_out[type_start[name]:type_start[name]+type_count[name]]) for name in type_name}
	print(type_acc,np.mean(acc_out))

	print(type_count)

	#保存到csv文件
	save_output = [[Path[i], acc_out[i]] for i in range(len(outputs_l))]		# Label[i], outputs_l[i],
	df_save = pd.DataFrame(save_output)
	df_save.to_csv(path_save,header=None,index=None)

	return


def load_image(Path = [], size = 512):
	'''
	return a np.array with 4 dims which has been normalization

	'''
	Images = []
	for path in Path:
		image = load_img(path = path, grayscale = True, 
			target_size = (size, size), interpolation='bilinear')		#'nearest'最邻近插值	双线性插值
		image = img_to_array(image)			#函数已经将图片扩展到3维度(320,320,1)
		Images.append(image)

	Images = np.asarray(Images).astype('float32')

	mean = np.mean(Images[:, :, :])			#normalization
	std = np.std(Images[:, :, :])
	Images[:, :, :] = (Images[:, :, :] - mean) / std
	return Images


def path_detail(Path):
	
	detail = [[],[],[],[],[]]			# train\XR_SHOULDER\patient00011\study1_positive\image1.png
	for path in Path:
		for i in range(len(detail)):
			[path,detail_v] = os.path.split(path)
			detail_v = detail_v.split('.')[0]
			detail[len(detail)-1-i].append(detail_v)

	label = []
	for study_str in detail[3]:
		if study_str.split('_')[-1]=='positive': #positive标签为1，否则为0；
			label+=[1]   #[1,1,1,1]    			#最后一级目录文件patient11880\\study1_negative\\image3.png
		else:				#negative
			label+=[0]

	detail.append(label)
	# [train_or_valid,studytype,patient_ID,study,im_name,label] = detail
	return detail


if __name__ == '__main__':


	# import argparse
	# parser = argparse.ArgumentParser(description='Run evaluate MURA experiment')
	# parser.add_argument('--path_load', type=str, default='../valid_labeled_studies.csv', help='Path to load csv')
	# parser.add_argument('--path_save', type=str, default='../valid_labeled_studies_output.csv', help='Path to save csv')
	# args = parser.parse_args()
	# print("file paths:")
	# for name, value in parser.parse_args()._get_kwargs():
	# 	print(name, value)
	# # for test 		../valid_labeled_studies.csv  ../valid_labeled_studies_output_test.csv


	path_load = 'F:/BaiduNetdiskDownload/MURA-v1.1/valid_labeled_studies.csv'
	path_save = 'F:/BaiduNetdiskDownload/MURA-v1.1/valid_labeled_studies_output.csv'
	path = [path_load, path_save]			# default path
	for i in range(1,len(sys.argv)):
		path[i-1] = sys.argv[i]
	evaluate_accuracy(path[0],path[1])
