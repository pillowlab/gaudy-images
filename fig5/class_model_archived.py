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
from scipy.linalg import eigh
from scipy.linalg import svd
from sklearn.decomposition import PCA
import gc

from numpy import linalg as LA

import pickle
import time


class ModelClass: 
 
	def __init__(self, num_models=25, learning_rate=1e-3, num_embed_vars=512, num_neurons=51):

		# model hyperparameters
		# learning_rate = 1e-3 #1e-1 #1e-2 #1e-3
		self.batch_size = 64
		self.num_models = num_models
		self.num_output_vars = 300
		self.num_neurons = num_neurons
		momentum = 0.7


		# define architecture
		x_input = Input(shape=(7,7,1024), name='feature_input')

		# accordian model (up to 512 then down to 256 then back up to 512)
		x = SeparableConv2D(filters=512, kernel_size=(1,1), strides=1, padding='same', name='initial_conv_layer')(x_input)

		x = SeparableConv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', name='layer1_conv')(x)
		x = BatchNormalization(axis=-1, name='layer1_bn')(x)
		x = Activation(activation='sigmoid', name='layer1_act')(x)

		x = SeparableConv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', name='layer2_conv')(x)
		x = BatchNormalization(axis=-1, name='layer2_bn')(x)
		x = Activation(activation='sigmoid', name='layer2_act')(x)

		x = SeparableConv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', name='layer3_conv')(x)
		x = BatchNormalization(axis=-1, name='layer3_bn')(x)
		x = Activation(activation='sigmoid', name='layer3_act')(x)
		
		# reshift matrices to take weighted average of spatial maps  
		x = DepthwiseConv2D(kernel_size=(7,7), strides=1, padding='valid', name='final_spatial_pool')(x)

		x = Flatten(name='embeddings')(x)
		x = Dense(units=self.num_output_vars, name='Beta')(x)   # linear readout layer

		base_model = Model(inputs=x_input, outputs=x)  # creates new network with smaller top network

		self.sgd = SGD(lr=learning_rate, decay=0., momentum=momentum, clipvalue=100.)

		base_model.compile(optimizer=self.sgd, loss='mean_squared_error')

		self.base_model = base_model

		self.embedding_model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('embeddings').output)

		self.save_folder = './results/saved_models/'

		# re-initialize networks to form an ensemble
		self.models_weights = []
		self.models_optimizer_states = []

		print('initializing model 0...')
		self.models_weights.append(self.base_model.get_weights())
		symbolic_optimizer_state = getattr(self.base_model.optimizer, 'weights')
		self.models_optimizer_states.append(K.batch_get_value(symbolic_optimizer_state))

		for imodel in range(1,self.num_models):
			print('initializing model ' + str(imodel) + '...')
			seed_index = 31490 + imodel
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


	def reset_Beta_weights_to_random(self):
		# resets all Betas to random, used for training new sessions
		# last one is intercepts, second to last weights, 512 x 300

		for imodel in range(self.num_models):
			num_filters = self.models_weights[imodel][-2].shape[0]

			self.models_weights[imodel][-1] = np.zeros(self.models_weights[imodel][-1].shape) # intercepts
			self.models_weights[imodel][-2][:,:self.num_neurons] = np.random.standard_normal(size=(num_filters,self.num_neurons))  # weights
			# self.models_weights[imodel][-2] = self.models_weights[imodel][-2] / (num_filters * self.num_neurons)

		self.curr_model = -1   # no matter what, reset the weights when training


	def train_models(self, features_train, responses_train, iepoch):

		num_samples = responses_train.shape[1]

		if iepoch == 0:
			self.num_neurons = responses_train.shape[0]
			self.reset_Beta_weights_to_random()

		# add padding to responses for unfilled units
		responses_train = np.vstack((responses_train, np.zeros(shape=(self.num_output_vars-self.num_neurons,num_samples))))
		
		if (self.is_first_batch == True):
			# train network one batch to retrieve optimizer states (we don't have them yet)
			#		NOTE --- this could make models dependent on each other, but assumption is this will have no effect
			#		b/c it's just one small batch
			self.base_model.train_on_batch(features_train[:64,:,:,:], responses_train[:,:64].T)
			self.set_initial_optimizer_states()
			self.is_first_batch = False

		# train models
		list_of_models_to_train = np.arange(self.num_models) 

		for imodel in list_of_models_to_train:

			# randomly shuffle training data
			r = np.random.permutation(num_samples)
			features_train = features_train[r,:,:,:]
			responses_train = responses_train[:,r]

			# swap model weights
			self.set_base_model(imodel)

			for ibatch in range(0,num_samples,self.batch_size):
				# if (ibatch + self.batch_size >= num_samples): # if going over the number of samples, skip
				# 	continue

				s = self.base_model.train_on_batch(features_train[ibatch:ibatch+self.batch_size,:,:,:], responses_train[:,ibatch:ibatch+self.batch_size].T)

			# update weights
			self.update_models_weights_list(imodel)


	def compute_validation_corr(self, features_val, responses_val):
		# compute corr over test data, averaged over neurons
		
		if self.num_neurons != responses_val.shape[0]:
			error('class_model.py: num neurons did not equal number of neurons in responses_val')

		responses_hat = self.get_predicted_ensemble_responses(features_val)
		
		corrs = np.diagonal(np.corrcoef(responses_hat, responses_val)[:self.num_neurons,self.num_neurons:])
		# corrs = corrs[np.isnan(corrs) != True]
		corr = np.mean(corrs)
		return corr


	def compute_test_frac_exp_var(self, features, responses1, responses2):
		# assumes new session, so Betas are not fit. need to fit them with ridge regression
		# also uses cross-validation

		num_images = features.shape[0]
		num_neurons = responses1.shape[0]

		# randomizes the heldout session...but this will likely also make test accuracy noisier
		# r = np.random.permutation(num_images)
		# features = np.copy(features[r])
		# responses1 = np.copy(responses1[:,r])
		# responses2 = np.copy(responses2[:,r])

		num_folds = 4

		responses_hat = np.zeros(responses1.shape)

		embeddings = []
		for imodel in range(self.num_models):
			embeds = self.get_embeddings(features, imodel)

			embeddings.append(embeds)

		num_test_images_per_fold = int(np.floor(num_images/num_folds))

		for ifold in range(num_folds):
			inds_test = np.arange(ifold*num_test_images_per_fold, (ifold+1)*num_test_images_per_fold)
			inds_train = np.array([x for x in range(num_folds * num_test_images_per_fold) if x not in inds_test])

			Ytrain = responses1[:,inds_train]
			responses_hat_over_models = np.zeros((self.num_models, num_neurons, inds_test.size))

			for imodel in range(self.num_models):
				Xtrain = embeddings[imodel][:,inds_train]
				Xtest = embeddings[imodel][:,inds_test]

				# alpha = np.minimum(4 * np.sum(np.var(Xtrain, axis=1)), 1e8)
				alpha = 0.01 * np.sum(np.var(Xtrain, axis=1))
				ridger = Ridge(alpha=alpha)

				ridger.fit(Xtrain.T, Ytrain.T)
				
				responses_hat_over_models[imodel,:,:] = ridger.predict(Xtest.T).T

			responses_hat[:,inds_test] = np.mean(responses_hat_over_models,axis=0)

		responses_hat = responses_hat[:,:-4] # remove last 4 samples in case not included in cross-validation
		responses1 = responses1[:,:-4]
		responses2 = responses2[:,:-4]

		frac_vars_pred = np.diagonal(np.corrcoef(responses_hat, responses1)[:num_neurons,num_neurons:])**2
		frac_vars_halfnhalf = np.diagonal(np.corrcoef(responses1, responses2)[:num_neurons,num_neurons:])**2

		return np.mean(frac_vars_pred / frac_vars_halfnhalf)


	def get_predicted_ensemble_responses(self, features, removePadding=True):
		# averages responses over ensemble

		num_images = features.shape[0]
		responses_avg = np.zeros((self.num_output_vars, num_images))
		for imodel in range(self.num_models):

			self.set_base_model(imodel)
			responses_avg += self.base_model.predict(features).T

		# remove padding
		if removePadding == True:
			responses_avg = responses_avg[:self.num_neurons,:]

		return responses_avg / self.num_models


	def get_predicted_responses_from_ith_model(self, features, imodel, remove_padding=True):
		# returns responses: (num_neurons, num_images)

		self.set_base_model(imodel)

		responses = self.base_model.predict(features).T

		# remove padding
		if remove_padding == True:
			responses = responses[:self.num_neurons,:]

		return responses


	def get_predicted_ensemble_responses_for_untrained_session(self, features, features_session, responses_session):
		# performs linear regression on features_session and responses_session, then predicts responses to features

		num_neurons = responses_session.shape[0]
		num_images = features.shape[0]

		# perform linear regression
		responses = np.zeros((self.num_models, num_neurons, num_images))
		for imodel in range(self.num_models):
			embeds = self.get_embeddings(features_session, imodel)
			alpha = 0.01 * np.sum(np.var(embeds, axis=1))
			ridger = Ridge(alpha=alpha)

			ridger.fit(embeds.T, responses_session.T)

			embeds = self.get_embeddings(features, imodel)
			responses[imodel,:,:] = ridger.predict(embeds.T).T

		return np.mean(responses,axis=0)


	def get_predicted_ensemble_responses_for_untrained_session_each_model(self, features, features_session, responses_session):
		# performs linear regression on features_session and responses_session, then predicts responses to features
		# returns responses (num_models, num_neurons, num_images)

		num_neurons = responses_session.shape[0]
		num_images = features.shape[0]

		# perform linear regression
		responses = np.zeros((self.num_models, num_neurons, num_images))
		for imodel in range(self.num_models):
			embeds = self.get_embeddings(features_session, imodel)
			alpha = 0.01 * np.sum(np.var(embeds, axis=1))
			ridger = Ridge(alpha=alpha)

			ridger.fit(embeds.T, responses_session.T)

			embeds = self.get_embeddings(features, imodel)
			responses[imodel,:,:] = ridger.predict(embeds.T).T

		return responses


	def get_embeddings(self, features, imodel):
		# returns embeddings: (num_embed_vars x num_images)

		self.set_base_model(imodel)
		embeddings = self.embedding_model.predict(features).T
		return embeddings


	def save_models(self, filetag='base_model', save_folder=None):
		# stores ensemble's weights for later use

		if save_folder == None:
			save_folder = self.save_folder

		self.base_model.save(save_folder + filetag + '.h5')
		with open(save_folder + 'models_weights_' + filetag + '.pkl', 'wb') as fp:
			pickle.dump(self.models_weights, fp)

		with open(save_folder + 'models_optimizer_states_' + filetag + '.pkl', 'wb') as fp:
			pickle.dump(self.models_optimizer_states, fp)


	def load_models(self, filetag='base_model', forTraining=False, load_folder=None):
		# loads ensemble's weights
		# forTraining flag allows different optimizers (so I can change step size between sessions, etc.)

		if load_folder == None:
			load_folder = self.save_folder

		self.base_model = load_model(load_folder + filetag + '.h5')
		with open(load_folder + 'models_weights_' + filetag + '.pkl', 'rb') as fp:
			self.models_weights = pickle.load(fp)

		with open(load_folder + 'models_optimizer_states_' + filetag + '.pkl', 'rb') as fp:
			self.models_optimizer_states = pickle.load(fp)

		self.embedding_model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('embeddings').output)

		# recompile in case step rate is different
		if forTraining == True:
			self.base_model.compile(optimizer=self.sgd, loss='mean_squared_error')

			self.is_first_batch = True   # already seen some data
			self.curr_imodel = -1  # unclear which is the current model

		if (self.num_models != len(self.models_weights)):
			raise ValueError('number of models does not equal the number of models in loaded model')
 




### NOTES


	# def compute_test_frac_exp_var(self, features, responses1, responses2):
	# 	# assumes new session, so Betas are not fit. need to fit them with ridge regression
	# 	# also uses cross-validation

	# 	num_images = features.shape[0]
	# 	num_neurons = responses1.shape[0]

	# 	# r = np.random.permutation(num_images)

	# 	# features = features[r]
	# 	# responses1 = responses1[:,r]
	# 	# responses2 = responses2[:,r]

	# 	num_folds = 4

	# 	responses_hat = np.zeros(responses1.shape)

	# 	embeddings = []
	# 	for imodel in range(self.num_models):
	# 		embeddings.append(self.get_embeddings(features, imodel))

	# 	num_test_images_per_fold = int(np.floor(num_images/num_folds))


	# 	for ifold in range(num_folds):
	# 		inds_test = np.arange(ifold*num_test_images_per_fold, (ifold+1)*num_test_images_per_fold)
	# 		inds_train = np.array([x for x in range(num_folds * num_test_images_per_fold) if x not in inds_test])

	# 		Ytrain = responses1[:,inds_train]
	# 		responses_hat_over_models = np.zeros((self.num_models, num_neurons, inds_test.size))

	# 		for imodel in range(self.num_models):
	# 			Xtrain = embeddings[imodel][:,inds_train]
	# 			Xtest = embeddings[imodel][:,inds_test]

	# 			# alpha = np.minimum(4 * np.sum(np.var(Xtrain, axis=1)), 1e8)
	# 			alpha = 0.01 * np.sum(np.var(Xtrain, axis=1))
	# 			ridger = Ridge(alpha=alpha)

	# 			ridger.fit(Xtrain.T, Ytrain.T)
				
	# 			responses_hat_over_models[imodel,:,:] = ridger.predict(Xtest.T).T

	# 		responses_hat[:,inds_test] = np.mean(responses_hat_over_models,axis=0)


	# 	responses_hat = responses_hat[:,:-4]
	# 	responses1 = responses1[:,:-4]
	# 	responses2 = responses2[:,:-4]

	# 	frac_vars_pred = np.diagonal(np.corrcoef(responses_hat, responses1)[:num_neurons,num_neurons:])**2
	# 	frac_vars_halfnhalf = np.diagonal(np.corrcoef(responses1, responses2)[:num_neurons,num_neurons:])**2

	# 	return frac_vars_pred / frac_vars_halfnhalf