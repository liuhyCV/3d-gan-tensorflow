import tensorflow as tf
import numpy as np
import time

from load_data import*

def conv3d(input_, output_dim, 
		k_d=5, k_h=5, k_w=5, d_d=2, d_h=2, d_w=2, stddev=0.02,
		name="conv3d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

def deconv3d(input_, output_shape,
		k_d=5, k_h=5, k_w=5, d_d=2, d_h=2, d_w=2, stddev=0.02,
		name="deconv3d", with_w=False):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
			initializer=tf.random_normal_initializer(stddev=stddev))

		deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape,
			strides=[1, d_d, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv

class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
			data_format='NHWC',
			decay=self.momentum, 
			updates_collections=None,
			epsilon=self.epsilon,
			scale=True,
			is_training=train,
			scope=self.name)

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
			tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
	if with_w:
		return tf.matmul(input_, matrix) + bias, matrix, bias
	else:
		return tf.matmul(input_, matrix) + bias

class GAN_3D(object):
	
	"""docstring for GAN_3D"""
	def __init__(self, sess, data_set_path, checkpoint_dir, 
		sample_g_path, dataset_name='default'):
		#super(GAN_3D, self).__init__()

		self.data_set_path = data_set_path
		self.checkpoint_dir = checkpoint_dir
		self.dataset_name = dataset_name
		self.sample_g_path = sample_g_path
		
		self.sess = sess
		self.input3d_depth = 64
		self.input3d_width = 64
		self.input3d_height = 64
		self.input3d_channels = 1

		self.epoch = 30
		self.batch_size = 100
		self.sample_input_data = 64

		self.z_dim = 200

		self.learning_rate1 = 0.025
		self.beta1 = 0.5

		self.learning_rate2 = 0.05
		self.beta2 = 0.5

		# batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')
		self.d_bn4 = batch_norm(name='d_bn4')
		self.d_bn5 = batch_norm(name='d_bn5')

		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')
		self.g_bn3 = batch_norm(name='g_bn3')

		self.build_model()


	def build_model(self):
		
		#create network model
		input_data_dims = [self.input3d_depth, self.input3d_height, 
				self.input3d_width, self.input3d_channels]

		self.input_data = tf.placeholder(
			tf.float32, [self.batch_size] + input_data_dims, name='real_3d_data')
		#self.sample_input_data = tf.placeholder(
			#tf.float32, [self.sample_num] + input_data_dims, name='fake_3d_data')
		
		self.z = tf.placeholder(
			tf.float32, [None, self.z_dim], name='z')
		self.z_sum = tf.histogram_summary("z", self.z)

		# three data flow
		self.G = self.generator(self.z)

		self.D_real, self.D_real_logits = self.discriminator(self.input_data,reuse=False)
		self.D_fake, self.D_fake_logits = self.discriminator(self.G,reuse=True)

		self.d_sum = tf.histogram_summary("d", self.D_real)
		self.d__sum = tf.histogram_summary("d_", self.D_fake)
		#self.G_sum = tf.image_summary("G", self.G)

		# loss function
		self.d_loss_real = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				self.D_real_logits, tf.ones_like(self.D_real)))
		self.d_loss_fake = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				self.D_fake_logits, tf.zeros_like(self.D_fake)))
		self.g_loss = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				self.D_fake_logits, tf.ones_like(self.D_fake)))

		self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()



	def train(self):

		#code for train GAN_3D model
		d_optim = tf.train.AdamOptimizer(self.learning_rate2, beta1=self.beta2) \
			.minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(self.learning_rate1, beta1=self.beta1) \
			.minimize(self.g_loss, var_list=self.g_vars)

		tf.global_variables_initializer().run()
		#tf.initialize_all_variables().run()

		self.g_sum = tf.merge_summary([self.z_sum, self.d__sum,
			self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = tf.merge_summary(
			[self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

		self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)


		#load 3d data
		list_3d_data_file = load_data_path(self.data_set_path, load_mode = 1)
		batch_idxs = len(list_3d_data_file) // self.batch_size

		counter = 1
		start_time = time.time()

		batch_data = np.zeros( (self.batch_size,64,64,64) )
		batch_images = np.zeros( (self.batch_size,64,64,64,1) )

		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		for epoch in range(0, self.epoch):
			for idx in range(0, batch_idxs):
				load_data_np( self.data_set_path,
						list_3d_data_file[idx*self.batch_size:(idx+1)*self.batch_size],
					batch_data )

				batch_images = batch_data.reshape(-1,64,64,64,1)

				batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
					.astype(np.float32)

				# Update D network
				_, summary_str = self.sess.run([d_optim, self.d_sum],
					feed_dict={ 
						self.input_data: batch_images,
						self.z: batch_z
						})
				self.writer.add_summary(summary_str, counter)

				# Update G network
				_, summary_str = self.sess.run([g_optim, self.g_sum],
					feed_dict={
						self.z: batch_z 
						})
				self.writer.add_summary(summary_str, counter)

				# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
				_, summary_str = self.sess.run([g_optim, self.g_sum],
					feed_dict={ 
						self.z: batch_z
						})
				self.writer.add_summary(summary_str, counter)

				errD_fake = self.d_loss_fake.eval({
					self.z: batch_z
					})
				errD_real = self.d_loss_real.eval({
					self.input_data: batch_images
					})
				errG = self.g_loss.eval({
					self.z: batch_z
					})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
				% (epoch, idx, batch_idxs,
				time.time() - start_time, errD_fake+errD_real, errG))
				

				if np.mod(counter,50) == 2:
					#print('test once')
					self.test( str(counter), 10 )

				if np.mod(counter, 500) == 2:
					self.save(self.checkpoint_dir, counter)


	def discriminator(self, image_3d, reuse=False):
		
		#discriminator network
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()
		
			h0 = lrelu(conv3d(image_3d, output_dim=64, name='d_h0_conv'))
			h1 = lrelu(self.d_bn1(conv3d(h0, output_dim=128, name='d_h1_conv')))
			h2 = lrelu(self.d_bn2(conv3d(h1, output_dim=256, name='d_h2_conv')))
			h3 = lrelu(self.d_bn3(conv3d(h2, output_dim=512, name='d_h3_conv')))
			h4 = self.d_bn4(conv3d(h3, output_dim=1, name='d_h4_conv'))

			return tf.nn.sigmoid(h4), h4

	def generator(self,z):
		
		#generator network
		with tf.variable_scope("generator") as scope:
			# project `z` and reshape
			self.z_, self.h0_w, self.h0_b = linear(
				z, 512*4*4*4, 'g_h0_lin', with_w=True)

			self.h0 = tf.reshape(
				self.z_, [-1, 4, 4, 4, 512])
			h0 = tf.nn.relu(self.g_bn0(self.h0))

			self.h1, self.h1_w, self.h1_b = deconv3d(
				h0, [self.batch_size, 8, 8, 8, 256], name='g_h1', with_w=True)
			h1 = tf.nn.relu(self.g_bn1(self.h1))

			h2, self.h2_w, self.h2_b = deconv3d(
				h1, [self.batch_size, 16, 16, 16, 128], name='g_h2', with_w=True)
			h2 = tf.nn.relu(self.g_bn2(h2))

			h3, self.h3_w, self.h3_b = deconv3d(
				h2, [self.batch_size, 32, 32, 32, 64], name='g_h3', with_w=True)
			h3 = tf.nn.relu(self.g_bn3(h3))

			h4, self.h4_w, self.h4_b = deconv3d(
				h3, [self.batch_size, 64, 64, 64, 1], name='g_h4', with_w=True)

			#return tf.nn.tanh(h4)
			return tf.nn.sigmoid(h4)


	def test(self, file_name, g_sample_num):

		mat_path = self.sample_g_path+'/'+file_name
		#print(mat_path)

		if not os.path.exists(mat_path):
			os.makedirs(mat_path)

		test_num = 10

		batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
			.astype(np.float32)
		generator_samples = self.sess.run(
			[self.G],
			feed_dict={
			self.z: batch_z
			}
		)
		generator_samples_np = np.array(generator_samples[0])

		#print(generator_samples_np[1,:,:,:].shape)
		for i_sample in range(0,self.batch_size):
			scipy.io.savemat(
				(mat_path+'/'+str(i_sample)+'.mat'), 
				{'instance':generator_samples_np[i_sample,:,:,:]} )


	def model_dir(self):
		return "{}_{}_{}".format(
			self.dataset_name, self.batch_size,
			64)

	def save(self, checkpoint_dir, step):
		model_name = "DCGAN.model"
		checkpoint_dir = checkpoint_dir+'/'+self.model_dir()

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
			checkpoint_dir+'/'+model_name,
			global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoints...")
		checkpoint_dir = checkpoint_dir+'/'+self.model_dir()

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, checkpoint_dir+'/'+ckpt_name)
			print(" [*] Success to read {}".format(ckpt_name))
			return True
		else:
			print(" [*] Failed to find a checkpoint")
			return False