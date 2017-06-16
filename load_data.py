
import numpy as np
import os
import re
import scipy.io


def load_data_np(train_data_path, list_train_data_name, batch_data):


	for i_data in range(0, len(list_train_data_name)):

		data_path = train_data_path + '/' + list_train_data_name[i_data]
		#data_path = 'D:/python_workspace/Liuhy/3Dporject/ModelNet40_Voxel/ModelNet40_Voxel/airplane/64/train/airplane_0001_1.mat'
		data_dict = scipy.io.loadmat(data_path)

		batch_data[i_data] = np.array(data_dict['instance'], dtype = int)


def load_data_path(train_data_path,load_mode):
	
	#load_mode==1:load one category
	#load_mode==2:load all category
	
	if(load_mode ==1):

		# find all datafile name in one category
		xxfiles = os.listdir(train_data_path)

		return xxfiles

	elif(load_mode==2):
		every_category_files_name = []
		path_category_3dmodel = []

		# find all category files name
		xxfiles = os.listdir(train_data_path)
		
		return xxfiles

	'''
	for i in range(0, len(xxfiles)):
		if( not re.search(r'[csv zip a-zA-Z]',xxfiles[i]) ):
			every_category_files_name.append(xxfiles[i])
	print(every_category_files_name)
	print(len(every_category_files_name))
	'''
