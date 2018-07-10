import os
import numpy as np
import cv2
import random


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
			else:
			    labels+=[0]
	print (len(Path))
	labels = np.asarray(labels)

	# print (Images[30],Images.shape[:])
	# #print (labels,Path) #测试效果
	# for i in range(10):
	# 	index_1 = random.randint(0,len(labels))
	# 	cv2.imshow( 'label = '+str(labels[index_1]),Images[index_1])
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	return Path, labels

def load_image(Path = '../valid/XR_ELBOW', size = 512):
	Images = []
	for path in Path:
		image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image,(size,size))
		image = randome_rotation_flip(image,size)
		Images.append(image)

	Images = np.asarray(Images).astype('float32')

	mean = np.mean(Images[:, :, :])			#normalization
	std = np.std(Images[:, :, :])
	Images[:, :, :] = (Images[:, :, :] - mean) / std
	Images = np.expand_dims(Images,axis=3)         #扩展维度3
	return Images





def randome_rotation_flip(image,size = 512):
	if random.randint(0,1):
		iamge = cv2.flip(image,1) # 1-->水平翻转 0-->垂直翻转 -1-->水平垂直

	if random.randint(0,1):
		angle = random.randint(-30,30)
		M = cv2.getRotationMatrix2D((size/2,size/2),angle,1)
		#第三个参数：变换后的图像大小
		image = cv2.warpAffine(image,M,(size,size))
	return image



if __name__ == '__main__':
	load_path()