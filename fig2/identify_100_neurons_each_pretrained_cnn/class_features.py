
# class to keep track of training/testing data for neural network architecture search
#
# Assumes:
#	Training set of 100,000 images
#	Validation set of 10k images
#	Test set of 10k images
#
# idea: keep training features in 5 buckets (10k each), and keep loading them
#  will need to keep track of which bucket is loaded, and how to call batches correctly

import os
from os import walk
import numpy as np

import tensorflow as tf

import sys
gpu_device = sys.argv[1]
print('using gpu ' + gpu_device)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device

from tensorflow.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet169
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

from tensorflow.keras import layers
from PIL import Image


class FeaturesClass:

 
### INIT FUNCTION

	def __init__(self):


		# do not load model on onset---b/c we have to call a bunch of classes with different models
		self.model = []



### CALL FUNCTIONS

	def load_model(self, pretrained_CNN=None):

		self.pretrained_CNN = pretrained_CNN

		del self.model

		if self.pretrained_CNN == 'VGG19':
			self.model = VGG19(weights='imagenet')
			layer_id = 'block4_pool'
		elif self.pretrained_CNN == 'InceptionV3':
			self.model = InceptionV3(weights='imagenet')
			layer_id = 'mixed4'
		elif self.pretrained_CNN == 'Densenet169':
			self.model = DenseNet169(weights='imagenet')
			layer_id = 'pool3_pool'
		elif self.pretrained_CNN == 'ResNet50':
			self.model = load_model('/usr/people/bcowley/adroit/code/synthAL/saved_models/resnet_model.h5')
			layer_id = 'activation_27'
		else:
			raise NameError('pretrained_CNN {:s} not recognized'.format(pretrained_CNN))

		x = AveragePooling2D(pool_size=(2,2), padding='valid')(self.model.get_layer(layer_id).output)
		x = Flatten()(x)

		self.model = Model(inputs=self.model.input, outputs=x)


	def get_features_from_imgs(self, imgs_raw):
		# assume imgs are preprocessed
		# Make sure you call load_model first!

		imgs = np.copy(imgs_raw)
		
		if self.pretrained_CNN == 'VGG19':
			imgs = preprocess_input_vgg19(imgs)
		elif self.pretrained_CNN == 'InceptionV3':
			imgs_processed = []
			for iimg in range(imgs.shape[0]):
				imgs_processed.append(np.array(Image.fromarray(imgs[iimg].astype('uint8')).resize((299,299))))

			imgs_processed = np.array(imgs_processed)
			imgs = preprocess_input_inceptionv3(imgs_processed)
		elif self.pretrained_CNN == 'Densenet169':
			imgs = preprocess_input_densenet169(imgs)
		elif self.pretrained_CNN == 'ResNet50':
			imgs = preprocess_input_resnet50(imgs)

		return self.model.predict(imgs)












