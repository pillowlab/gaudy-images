# basic structure:
#	for ensemble of models
#	have one base model, and lists of ensemble weights
#	keep swapping out weights, training base model, and update lists

import numpy as np
import tensorflow as tf

import sys

gpu_device = sys.argv[1]
print('using gpu ' + gpu_device)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device


from tensorflow.keras import backend as K
config = tf.ConfigProto()
# config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=2, allow_soft_placement=True)
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
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import DepthwiseConv2D

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from numpy.random import seed
from tensorflow import set_random_seed
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import clone_model --- later version than what I have

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from scipy.linalg import eigh
from scipy.linalg import svd
from sklearn.decomposition import PCA
import gc

from numpy import linalg as LA

import pickle
import time


class ModelEnsembleClass: 

	def __init__(self, num_models=10, num_output_vars=100, learning_rate=1e0, seed_index=31490):

		# model hyperparameters
		# learning_rate = 1e0
		momentum = 0.7
		self.batch_size = 64
		self.num_models = num_models
		self.num_output_vars = num_output_vars


		# define architecture
		x_input = Input(shape=(7,7,1024), name='feature_input')

		# accordian model (up to 512 then down to 256 then back up to 512)
		x = SeparableConv2D(filters=512, kernel_size=(1,1), strides=1, padding='same', name='initial_conv_layer')(x_input)

		x = SeparableConv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', name='layer1_conv')(x)
		x = BatchNormalization(axis=-1, name='layer1_bn')(x)
		x = Activation(activation='relu', name='layer1_act')(x)

		x = SeparableConv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', name='layer2_conv')(x)
		x = BatchNormalization(axis=-1, name='layer2_bn')(x)
		x = Activation(activation='relu', name='layer2_act')(x)

		x = SeparableConv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', name='layer3_conv')(x)
		x = BatchNormalization(axis=-1, name='layer3_bn')(x)
		x = Activation(activation='relu', name='layer3_act')(x)
		
		# reshift matrices to take weighted average of spatial maps  
		x = DepthwiseConv2D(kernel_size=(7,7), strides=1, padding='valid', name='final_spatial_pool')(x)

		x = Flatten(name='embeddings')(x)
		x = Dense(units=self.num_output_vars, name='Beta')(x)   # linear readout layer

		base_model = Model(inputs=x_input, outputs=x)  # creates new network with smaller top network

		self.sgd = SGD(lr=learning_rate, decay=0., momentum=momentum, clipvalue=100.)

		base_model.compile(optimizer=self.sgd, loss='mean_squared_error')

		self.base_model = base_model

		self.save_folder = './results/saved_models/'

		# self.base_model.summary()

		# re-initialize networks to form an ensemble
		self.models_weights = []
		self.models_optimizer_states = []

		print('initializing model 0...')
		self.models_weights.append(self.base_model.get_weights())
		symbolic_optimizer_state = getattr(self.base_model.optimizer, 'weights')
		self.models_optimizer_states.append(K.batch_get_value(symbolic_optimizer_state))


		for imodel in range(1,self.num_models):
			print('initializing model ' + str(imodel) + '...')
			seed_index = seed_index + imodel
			seed(seed_index)
			set_random_seed(seed_index)
			session = K.get_session()
			for layer in self.base_model.layers:
				for v in layer.__dict__.values():
					if hasattr(v, 'initializer'):
						v.initializer.run(session=session)

			self.models_weights.append(self.base_model.get_weights())
			symbolic_optimizer_state = getattr(self.base_model.optimizer, 'weights')
			self.models_optimizer_states.append(K.batch_get_value(symbolic_optimizer_state))

		self.is_first_batch = True # keeps track of when the very first batch is run
		self.curr_imodel = -1


	def set_base_model(self, imodel):
		# load imodel's weights into base_model

		# if imodel == self.curr_imodel, do nothing, already at hand
		if imodel != self.curr_imodel:  # not the same, so you need to change the model
			self.base_model.set_weights(self.models_weights[imodel])
			self.base_model.optimizer.set_weights(self.models_optimizer_states[imodel])
			self.curr_imodel = imodel


	def update_models_weights_list(self, imodel):
		# take base_model's weights and store them (updating stored weights)

		self.models_weights[imodel] = self.base_model.get_weights()
		symbolic_optimizer_state = getattr(self.base_model.optimizer, 'weights')
		self.models_optimizer_states[imodel] = K.batch_get_value(symbolic_optimizer_state)


	def set_initial_optimizer_states(self):
		# need to run one batch before optimizer is initialized, so now re-initialize all models' states

		symbolic_optimizer_state = getattr(self.base_model.optimizer, 'weights')
		for imodel in range(self.num_models):
			self.models_optimizer_states[imodel] = K.batch_get_value(symbolic_optimizer_state)


	def compute_corr(self, features_test, responses_test):
		# compute corr over test data, averaged over neurons
		
		num_neurons = responses_test.shape[0]

		responses_hat = self.get_predicted_ensemble_responses(features_test)

		corr = np.mean(np.diagonal(np.corrcoef(responses_hat, responses_test)[:num_neurons,num_neurons:]))
		return corr


	def train_models(self, features_train, responses_train, imodel=-1):

		num_samples = responses_train.shape[1]

		if (self.is_first_batch == True):
			# train network one batch to retrieve optimizer states (we don't have them yet)
			#		NOTE --- this could make models dependent on each other, but assumption is this will have no effect
			#		b/c it's just one small batch
			self.base_model.train_on_batch(features_train[:64,:,:,:], responses_train[:,:64].T)
			self.set_initial_optimizer_states()
			self.is_first_batch = False

		# train models
		if imodel == -1:  # train all models
			list_of_models_to_train = np.arange(self.num_models) 
		else:  # train only one specified model
			list_of_models_to_train = [imodel]

		for imodel in list_of_models_to_train:

			# randomly shuffle training data
			r = np.random.permutation(num_samples)
			features_train = features_train[r,:,:,:]
			responses_train = responses_train[:,r]

			# swap model weights
			self.set_base_model(imodel)

			for ibatch in range(0,num_samples,self.batch_size):
				if (ibatch + self.batch_size >= num_samples): # if going over the number of samples, skip
					continue

				s = self.base_model.train_on_batch(features_train[ibatch:ibatch+self.batch_size,:,:,:], responses_train[:,ibatch:ibatch+self.batch_size].T)

			# update weights
			self.update_models_weights_list(imodel)


	def get_predicted_ensemble_responses(self, features):
		# averages responses over ensemble

		num_images = features.shape[0]
		responses_avg = np.zeros((self.num_output_vars, num_images))
		for imodel in range(self.num_models):

			self.set_base_model(imodel)
			responses_avg += self.base_model.predict(features).T

		return responses_avg / self.num_models


	def get_predicted_responses_from_ith_model(self, features, imodel):

		self.set_base_model(imodel)

		return self.base_model.predict(features).T



	def save_models(self, filetag='base_model', save_folder=None):
		# stores ensemble's weights for later use

		if save_folder == None:
			save_folder = self.save_folder

		self.base_model.save(save_folder + filetag + '.h5')
		with open(save_folder + 'models_weights_' + filetag + '.pkl', 'wb') as fp:
			pickle.dump(self.models_weights, fp)

		with open(save_folder + 'models_optimizer_states_' + filetag + '.pkl', 'wb') as fp:
			pickle.dump(self.models_optimizer_states, fp)



	def load_models(self, filetag='base_model', save_folder=None):
		# loads ensemble's weights

		if save_folder == None:
			save_folder = self.save_folder

		self.base_model = load_model(save_folder + filetag + '.h5')
		with open(save_folder + 'models_weights_' + filetag + '.pkl', 'rb') as fp:
			self.models_weights = pickle.load(fp)

		with open(save_folder + 'models_optimizer_states_' + filetag + '.pkl', 'rb') as fp:
			self.models_optimizer_states = pickle.load(fp)

		self.is_first_batch = False   # already seen some data
		self.curr_imodel = -1  # unclear which is the current model

		if (self.num_models != len(self.models_weights)):
			raise ValueError('number of models does not equal the number of models in loaded model')
 




