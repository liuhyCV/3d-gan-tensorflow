This repository contains the reproduce codes for the paper [Learning a Probabilistic Latent Space of Object Shapes 
via 3D Generative-Adversarial Modeling](http://3dgan.csail.mit.edu/). 


# Prerequisites
### Matlab
here use Matlab code to process ShapeNetCore/ModelNet10/ModelNet40 dataset, convert .off type to voxel type and save it in .mat file
### Tensorflow
3d-gan network is being build and train using tensorflow r-0.12

# Guide

### main.py
run the python file. It will start build the 3D-GAN network and start training

### load_data.py
load_data.py contains two functions: load_data_np, load_data_path, which is used to load .mat file and generate train files list

### 3dgan_model.py
3dgan_model.py contains class GAN_3D, which can build the network and do training.
this file is modified based on [DCGAN in Tensorflow](https://github.com/carpedm20/DCGAN-tensorflow), by changing input and filters size and dim.


# Note
For some reasons, this repository is unfinished and if you are interesting in it, please contact me and work together
