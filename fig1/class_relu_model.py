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

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input


from tensorflow.keras.optimizers import SGD

from numpy.random import seed



class ReluModel:

### INTERNAL FUNCTIONS

	def get_model(self):

		x_input = Input(shape=(112,112), name='input')

		x = Flatten()(x_input)
		x = Dense(units=1, name='Beta')(x)
		x = Activation(activation='relu')(x)

		return Model(inputs=x_input, outputs=x)



### INIT FUNCTION
	def __init__(self):

		learning_rate = 3e-7
		momentum = 0.99
		self.batch_size = 64

		self.model = self.get_model()

		sgd = SGD(lr=learning_rate, decay=0., momentum=momentum, clipvalue=100.)

		self.model.compile(optimizer=sgd, loss='mean_squared_error')



### EXTERNAL FUNCTIONS

	def compute_frac_var(self, imgs_test, responses_test):
		# compute frac_var over test data, averaged over neurons
		
		responses_hat = self.get_predicted_responses(imgs_test)

		corr = np.corrcoef(responses_hat, responses_test)[0,1]

		return corr**2 # returns Rsquared


	def train_model(self, imgs_train, responses_train):

		num_samples = responses_train.size
		r = np.random.permutation(num_samples)

		imgs_train = imgs_train - 128
		for ibatch in range(0, num_samples, self.batch_size):
			if (ibatch + self.batch_size >= num_samples):  # don't go over num_samples
				continue

			self.model.train_on_batch(imgs_train[r[ibatch:ibatch+self.batch_size],:,:], 
								responses_train[r[ibatch:ibatch+self.batch_size]])


	def get_predicted_responses(self, imgs):

		imgs = (imgs - 128)
		return np.squeeze(self.model.predict(imgs))


	def get_kernel(self):

		 weights = self.model.get_weights()[0]

		 weights = np.reshape(weights, (112,112))
		 return weights


	def initialize_weights(self, seed_index):
		# ensures random_selection and gaudi images start with same network weights
		weights = self.model.get_layer('Beta').get_weights()

		seed(seed_index)
		weights[0] = np.random.uniform(low=-1., high=1., size=(weights[0].shape))/128.
		self.model.get_layer('Beta').set_weights(weights)
