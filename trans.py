#import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import scipy.io

root_file = 'D:/python_workspace/Liuhy/3Dporject/3DShapeNets/3DGan_liuhy/'
train_data_file = root_file + 'volumetric_data/' + 'airplane/30/train'

for root_temp, sub_dirs_temp, files_temp in os.walk(train_data_file):
	#do nothing
root = root_temp
sub_dirs = sub_dirs_temp
files = files_temp

print(root)
print(sub_dirs)
print(len(files))

data = np.zeros( (len(files),30,30,30) )
print(data.shape)

mat_number = 0

for special_file in files[0:len(files)]:
	spcial_file_dir = os.path.join(root, special_file)
	print(special_file)

	data_dict = scipy.io.loadmat(spcial_file_dir)
	data[mat_number] = data_dict['instance']
	
	mat_number = mat_number + 1
	#print(type(data))
	
	#print(type(data_mat))

print(mat_number)

np.save("airplane_3d_data.npy", data)


'''
data_plot = np.zeros( (len(files),3,30*30*30) )



for i_x in range(0,30):
	for i_y in range(0,30):
		for i_z in range(0,30):
			if data[0][i_x][i_y][i_z]>0 :
				data_plot[0][0][i_x*30*30+i_y*30+i_z]=i_x
				data_plot[0][1][i_x*30*30+i_y*30+i_z]=i_y
				data_plot[0][2][i_x*30*30+i_y*30+i_z]=i_z


x,y,z = data_plot[0][0],data_plot[0][1],data_plot[0][2]
ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程

#将数据点分成三部分画，在颜色上有区分度
ax.scatter(x,y,z,c='y') #绘制数据点

ax.set_zlabel('Z') #坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')

ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 30)

plt.show()


'''