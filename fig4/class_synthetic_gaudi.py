# class to generate images from generator network (trained on Gaudi images)
#
# maximizes ensemble variance, where gradient is in feature space, but 

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import losses as losses

from tensorflow.keras.models import Model

import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from numpy import linalg as LA
import scipy.spatial.distance as distance

from sklearn.metrics import pairwise_distances

import scipy.ndimage as ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
from numpy.random import seed



class SyntheticGaudiClass:


### INTERNAL HELPER FUNCTIONS

	def initialize_loss_and_grad_funcs(self, F, R, M):

		# generate loss and grad generator resnet and sister network
		#	features --> generator --> resnet --> sisternet --> responses
			model_input = self.generator_resnet_sister_model.inputs[0]
			model_output = self.generator_resnet_sister_model.output

			y_true = K.placeholder((100,))
			loss = losses.mean_squared_error(model_output, y_true)
			grad = K.gradients(loss, model_input)[0]

			self.loss_func_gen_res_sis_model = K.function([model_input, y_true], [loss])
			self.grad_func_gen_res_sis_model = K.function([model_input, y_true], [grad])

		# generate loss and grad generator resnet
			model_input = self.generator_resnet_model.inputs[0]
			model_output = self.generator_resnet_model.output

			y_true = K.placeholder((1,7,7,1024))
			loss = K.mean(K.square(model_output - y_true))
			grad = K.gradients(loss, model_input)[0]

			self.loss_func_generator_resnet = K.function([model_input, y_true], [loss])
			self.grad_func_generator_resnet = K.function([model_input, y_true], [grad])


	def clip_features(self, features):
		# set feature values >= 0
		# note: feature values should not be negative (b/c last layer of ResNet is relu's)
		
		features = np.clip(features, a_min=0., a_max=None)

		return features


	def get_image_from_features(self, I, features):
		# assumes features (num_samples x 1024 x 7)
		# returns (num_pixels x num_pixels x 3) image in raw format

		img = self.generator_model.predict(features)[0]  # generator returns image in resnet format

		img = I.reverse_preprocessing(img)  # reverses resnet format to raw image format
		
		img = I.get_gaudi_image(img)

		return img


	def compute_ensemble_variance(self, M, features):

		responses = np.zeros(shape=(M.num_models, M.num_output_vars))
		for imodel in range(M.num_models):
			responses[imodel,:] = M.get_predicted_responses_from_ith_model(features, imodel)[:,0]

		return np.sum(np.var(responses,axis=0))


	def get_nearest_neighbor_features(self, feature_synth):
		# nearest neighbor only computed in feature space...
		# dists = pairwise_distances(X=np.reshape(feature_synth, (1,feature_synth.size)), Y=np.asarray(self.features), n_jobs=4)
		dists = distance.cdist(XA=np.reshape(feature_synth, (1,feature_synth.size)), XB=np.asarray(self.features))
		ind_min = np.argmin(dists)
	
		return ind_min


	def get_gradient_ensvar(self, I, R, M, feature_synth):

		# randomly choose two models
			r = np.random.permutation(M.num_models)
			ind_ith_model = r[0]
			ind_jth_model = r[1]

		# compute gradient for ith model
			y_true = np.squeeze(M.get_predicted_responses_from_ith_model(feature_synth, ind_jth_model))
			M.set_base_model(ind_ith_model)
			loss_ith_model = np.sqrt(self.loss_func_gen_res_sis_model([feature_synth, y_true])[0])
			grad_ith_model = 1. / 2. / loss_ith_model * self.grad_func_gen_res_sis_model([feature_synth, y_true])[0]

		# compute gradient for jth model
			y_true = np.squeeze(M.get_predicted_responses_from_ith_model(feature_synth, ind_ith_model))
			M.set_base_model(ind_jth_model)
			loss_jth_model = np.sqrt(self.loss_func_gen_res_sis_model([feature_synth, y_true])[0])
			grad_jth_model = 1. / 2. / loss_jth_model * self.grad_func_gen_res_sis_model([feature_synth, y_true])[0]

		# include diversity term
			if (len(self.features) > 0):
				ind_min = self.get_nearest_neighbor_features(feature_synth)
				y_true = np.reshape(self.features[ind_min], tuple(feature_synth.shape))

				loss_features = np.sqrt(self.loss_func_generator_resnet([feature_synth, y_true])[0])
				grad_features = 1. / 2. / loss_features * self.grad_func_generator_resnet([feature_synth, y_true])[0]
			else:
				loss_features = -1.
				grad_features = 0.

		# compute and return gradient
			grad = grad_ith_model + grad_jth_model + self.lambda_featuresNN * grad_features
					  # sum gradients (which works out to df/dx - dr/dx)
		

		# TESTING:
			grad /= np.sqrt(np.sum(grad**2))

			return grad  


	def constrain_grad_features(self, I, feature_synth):
		# constrain update to be within features range
			feature_synth = self.clip_features(feature_synth) # ensure feature values are nonnegative
			
			return feature_synth


	def get_gradient_update(self, I, R, M, feature_synth):

		grad = self.get_gradient_ensvar(I, R, M, feature_synth)		

		feature_synth += self.step_size * grad

		feature_synth = self.constrain_grad_features(I, feature_synth)
		
		return feature_synth


	def synthesize_images(self, I, R, M):

		imgs = []
		self.features = []

		for iimg in range(self.num_cand_images):

			print('   img ' + str(iimg))
			
			# get initial image
			seed()
			
			img_synth = I.get_random_natural_images()   # returns 1 x 224 x 224 x 3 image
			img_synth = I.get_gaudi_images(img_synth)  # returns 1 x 224 x 224 x 3 image
			feature_synth = self.F.get_features_from_imgs(img_synth)

			# ens_var_before = self.compute_ensemble_variance(M, feature_synth)

			# perform gradient ascent to maximize ensemble variance
			for iepoch in range(self.num_epochs):

				feature_synth = self.get_gradient_update(I, R, M, feature_synth)

			img = self.get_image_from_features(I, feature_synth)  # returns 224 x 224 x 3 image
			imgs.append(img)  # gets image from feature_synth

			self.features.append(np.reshape(feature_synth, (feature_synth.size,)))
			
			# ens_var_after = self.compute_ensemble_variance(M, feature_synth)
			# print('before: {:f}, after: {:f}'.format(ens_var_before, ens_var_after))

		return np.asarray(imgs)



### INIT FUNCTION

	def __init__(self, I, F, R, M, num_rands=250, num_epochs=20, step_size=2e1, lambda_featuresNN=1e-6):
			#  num_rands=250, num_epochs=50, step_size=500, lambda_featuresNN=10.
		self.F = F

		# load natural prior model
		filepath_generator_model = './results/saved_models/generator_networks/generator_gaudi.h5'
		self.generator_model = load_model(filepath_generator_model)

		self.resnet_model = F.resnet_model
		self.sister_model = M.base_model

		self.generator_resnet_model = Model(inputs=self.generator_model.input, outputs=self.resnet_model(self.generator_model.output))
		self.generator_resnet_sister_model = Model(inputs=self.generator_model.input, outputs=self.sister_model(self.resnet_model(self.generator_model.output)))

		self.initialize_loss_and_grad_funcs(F, R, M)

		# hyperparameters
		self.num_rands = num_rands
		self.num_cand_images = 500 - self.num_rands
		self.step_size = step_size
		self.num_epochs = num_epochs

		self.lambda_featuresNN = lambda_featuresNN



### EXTERNAL CALL FUNCTIONS

	def get_images_to_show(self, I, F, R, M):

		if self.num_cand_images > 0:
			imgs_chosen = self.synthesize_images(I, R, M)

		if self.num_rands > 0:
			imgs_rand = I.get_random_natural_images(self.num_rands)

		if self.num_rands == 500:
			imgs_chosen = imgs_rand
		elif self.num_rands > 0:
			imgs_chosen = np.concatenate((imgs_chosen, imgs_rand), axis=0)

		return imgs_chosen



### NOTES







