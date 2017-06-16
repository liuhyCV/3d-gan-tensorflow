import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import scipy.io

#import mayavi.mlab

root_file = 'D:/python_workspace/Liuhy/3Dporject/3DShapeNets/3DGan_liuhy/sample_generator/'
train_data_file = root_file

voxel_data_len = 64


data = np.zeros( (1,voxel_data_len,voxel_data_len,voxel_data_len) )
print(data.shape)



data_dict = scipy.io.loadmat(root_file+ '52/' + '20.mat')
#print(data_dict)
#print(data_dict.shape)
#data[0] = data_dict['instance']
#print(data_dict['instance'][:,:,:,0].shape)
data[0] = data_dict['instance'][:,:,:,0]

data_plot = np.zeros( (1,3,voxel_data_len*voxel_data_len*voxel_data_len) )

for i_x in range(0,voxel_data_len):
	for i_y in range(0,voxel_data_len):
		for i_z in range(0,voxel_data_len):
			if data[0][i_x][i_y][i_z]>0.8 :
				data_plot[0][0][i_x*(voxel_data_len^2)+i_y*voxel_data_len+i_z]=i_x
				data_plot[0][1][i_x*(voxel_data_len^2)+i_y*voxel_data_len+i_z]=i_y
				data_plot[0][2][i_x*(voxel_data_len^2)+i_y*voxel_data_len+i_z]=i_z

x,y,z = data_plot[0][0],data_plot[0][1],data_plot[0][2]


ax=plt.subplot(111,projection='3d')

ax.scatter(x,y,z,c='y')

ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')

ax.set_xlim(0, voxel_data_len)
ax.set_ylim(0, voxel_data_len)
ax.set_zlim(0, voxel_data_len)

plt.show()
