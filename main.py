import tensorflow as tf
import numpy as np
import time

from Gan_3D_model import*




with tf.Session() as sess:

	train_data_path = 'D:/python_workspace/Liuhy/3Dporject/ModelNet40_Voxel/ModelNet40_Voxel/airplane/64/train'
	checkpoint_dir = 'D:/python_workspace/Liuhy/3Dporject/3DShapeNets/3DGan_liuhy/checkpoints'
	sample_g_path = 'D:/python_workspace/Liuhy/3Dporject/3DShapeNets/3DGan_liuhy/sample_generator'
	
	gan3d = GAN_3D(sess = sess, data_set_path = train_data_path, 
		checkpoint_dir = checkpoint_dir, sample_g_path = sample_g_path)

	gan3d.train()