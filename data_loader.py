# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import random
import keras.backend as K

def load_path(root_path = '../valid/XR_ELBOW', size = 512):
	'''
	load MURA data

	'''
	Path = []
	labels = []
	for root,dirs,files in os.walk(root_path): #读取所有图片, os.walk返回迭代器genertor 遍历所有文件
		for name in files:
			path_1 = os.path.join(root,name)
			Path.append(path_1)
			if root.split('_')[-1]=='positive': #positive标签为1，否则为0；
				labels+=[1]   #[1,1,1,1]    			#最后一级目录文件patient11880\\study1_negative\\image3.png
			else:				#negative
				labels+=[0]

	count = [labels.count(0),labels.count(1)]			#count of negative,positive
	# print (len(Path))
	# print('[N,P]',count)

	# 增加positive数据，使两者相等,positive 少于negative
	len_label = len(labels)
	if count[0] > count[1]:
		len_app = count[0]-count[1]
		lab_app = 1
	else:
		len_app = count[1]-count[0]
		lab_app = 0
	for i in range(len_app):
		idx = random.randint(0,len_label-1)
		while labels[idx]!=lab_app:
			idx = random.randint(0,len_label-1)
		Path.append(Path[idx])
		labels+=[lab_app]
		count[lab_app]+=1
	# print('[N,P]',count)
	
	# detail = path_detail(Path) #路径细节
	labels = np.asarray(labels)

	# print (Images[30],Images.shape[:])
	# #print (labels,Path) #测试效果
	# for i in range(10):
	# 	index_1 = random.randint(0,len(labels))
	# 	cv2.imshow( 'label = '+str(labels[index_1]),Images[index_1])
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	return Path, labels

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



def load_image(Path = ['test_images/image1.png'], size = 320, rotation = True, normalization = True):
	'''
	return a np.array with 4 dims which has been normalization

	'''
	Images = []
	for path in Path:
		image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		hight, width = image.shape
		if hight > width:
			hight_target = size
			width_target = int(width * (hight_target/hight))
		else:
			width_target = size
			hight_target = int(hight * (width_target/width))
		image = cv2.resize(image,(width_target,hight_target))

		# 固定原图长宽比例，减少对图片的破坏，缩放图片到要求size，并空余处补0
		Image = np.zeros((size,size), np.uint8)  
		start_h = int((size-hight_target)/2)
		start_w = int((size-width_target)/2)
		Image[start_h:start_h+hight_target,start_w:start_w+width_target] = image

		if rotation:
			Image = randome_rotation_flip(Image,size)
		# cv2.imshow(path,Image)
		# cv2.waitKey()
		Images.append(Image)
	Images = np.asarray(Images)

	if normalization:									#normalization
		Images.astype('float32')
		mean = np.mean(Images[:, :, :])			
		std = np.std(Images[:, :, :])
		Images[:, :, :] = (Images[:, :, :] - mean) / std

	if K.image_data_format() == "channels_first":
		Images = np.expand_dims(Images,axis=1)		   #扩展维度1
	if K.image_data_format() == "channels_last":
		Images = np.expand_dims(Images,axis=3)         #扩展维度3
	return Images





def randome_rotation_flip(image=None,size = 512):
	if random.randint(0,1):
		iamge = cv2.flip(image,1) # 1-->水平翻转 0-->垂直翻转 -1-->水平垂直

	if random.randint(0,1):
		angle = random.randint(-30,30)
		M = cv2.getRotationMatrix2D((size/2,size/2),angle,1)
		#第三个参数：变换后的图像大小
		image = cv2.warpAffine(image,M,(size,size))
	return image



if __name__ == '__main__':
	# load_path()
	load_image()
