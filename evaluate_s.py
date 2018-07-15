# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import csv
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, array_to_img

def evaluate(path_load = '', path_save = ''):				

	im_size = 320  

	with open(path_load,'r') as load_file:
		rows = csv.reader(load_file)
		Path = [row[0] for row in rows]

	Path = [os.path.split(path)[0]+'/' for path in Path]
	Path = list(set(Path))
	Path.sort()

	# 'load pretrained model '
	pretrained_model_path = 'src/MURA_model_dense169.h5'
	model = load_model(pretrained_model_path)	

	outputs = [0]*len(Path)
	for idx,path in enumerate(Path):			
		im_paths = os.listdir(path)
		im_paths = [os.path.join(path,im_path) for im_path in im_paths]
		im = load_image(Path=im_paths, size=im_size)   
		predictions = model.predict(im)
		outputs[idx] = float(np.mean(predictions))
	outputs_l = [int(round(outputs[i])) for i in range(len(outputs))]	

	with open(path_save,'w') as save_file:			# newline=''
		csv_writer = csv.writer(save_file,lineterminator='\n')		# lineterminator='\n'
		csv_writer.writerows([[path,output] for path,output in zip (Path,outputs_l)])
	return


def load_image(Path = ['MURA-v1.1/DenseNet_v1/test_images/image1.png'], size = 320):
	'''
	return a np.array with 4 dims which has been normalization

	'''
	Images = []
	for path in Path:
		image = load_img(path = path, grayscale = True, 
			target_size = None, interpolation='bilinear')		
		image_a = img_to_array(image)	
		hight, width, channels = image_a.shape
		# print(hight,width,channels,image.size)
		if hight > width:
			hight_target = size
			width_target = int(width * (hight_target/hight))
		else:
			width_target = size
			hight_target = int(hight * (width_target/width))
		# image = array_to_img(image_a).resize((width_target,hight_target),'bilinear')
		image = load_img(path = path, grayscale = True, 
			target_size = (hight_target,width_target), interpolation='bilinear')	
		# print(image.size)
		image_a = img_to_array(image)

		# 固定原图长宽比例，减少对图片的破坏，缩放图片到要求size，并空余处补0
		Image = np.zeros((size,size,channels), np.uint8)  
		start_h = int((size-hight_target)/2)
		start_w = int((size-width_target)/2)
		Image[start_h:start_h+hight_target,start_w:start_w+width_target,:] = image_a	
		# import cv2
		# cv2.imshow(path,Image)
		# cv2.waitKey()	
		Images.append(Image)
	Images = np.asarray(Images).astype('float32')

	mean = np.mean(Images[:, :, :])			#normalization
	std = np.std(Images[:, :, :])
	Images[:, :, :] = (Images[:, :, :] - mean) / std
	return Images


if __name__ == '__main__':

	path_load = 'F:/BaiduNetdiskDownload/MURA-v1.1/valid_image_paths.csv'
	path_save = 'F:/BaiduNetdiskDownload/MURA-v1.1/valid_labeled_studies_output.csv'

	path = [path_load, path_save]			# default path
	for i in range(1,len(sys.argv)):
		path[i-1] = sys.argv[i]

	# evaluate(path[0],path[1])
	load_image()
