# change number of gaudi images vs random images (total sum is 500 per session)


import numpy as np
import sys

import class_model_ensemble
import class_images
import class_features
import class_surrogate_responses

import class_random_selection
import class_gaudi
import class_largebank_normal_only
import class_largebank_gaudi_only
import class_largebank_gaudi_and_normal

from numpy.random import seed
from tensorflow import set_random_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


### HELPER FUNCTIONS

def get_test_set(I, F, R):

	imgs = I.get_random_natural_images(num_random_images=4000)

	features_test = F.get_features_from_imgs(imgs)
	responses_test = R.get_responses_from_imgs(imgs)

	return features_test, responses_test


def get_initial_train_set(I, F, R):
	# get small training set to set parameters
	imgs = I.get_random_natural_images(num_random_images=64)

	features_test = F.get_features_from_imgs(imgs)
	responses_test = R.get_responses_from_imgs(imgs)

	return features_test, responses_test


def run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_ensemble_networks, num_sessions, irun=0):

	corrs_test = []

	for isession in range(num_sessions):

		imgs_train = A.get_images_to_show(I, F, R, M)

		features_train = F.get_features_from_imgs(imgs_train)
		responses_train = R.get_responses_from_imgs(imgs_train)

		print('session: ' + str(isession))
		print('   num chosen images: {:d}'.format(imgs_train.shape[0]))

		for iepoch in range(num_epochs):

			M.train_models(features_train, responses_train)

			corr_train = M.compute_corr(features_train, responses_train)

			corr_val = M.compute_corr(features_test, responses_test)

			print('         epoch {:d}: corr_val = {:0.4f}, corr_train = {:0.4f}'.format(iepoch, 
																corr_val, corr_train))

		corr_test = corr_val  # same computation
		corrs_test.append(corr_test)

		print('{:s}: corr_test = {:0.4f}'.format(AL_algo, corr_test))

		np.save('./results/corrs_test_' + pretrained_CNN + '_' + AL_algo + '_' + str(num_ensemble_networks) + 'networks_run{:d}.npy'.format(irun), np.asarray(corrs_test))



### MAIN SCRIPT

# python -u script_num_ensemble_networks.py #gpu_id

pretrained_CNN = 'VGG19'  

num_sessions = 20
num_epochs = 5
num_models = 1

seed_index = 150;

# get models set up
seed(seed_index)
set_random_seed(seed_index)

learning_rate = 1e-1
	
M = class_model_ensemble.ModelEnsembleClass(num_models=num_models, num_output_vars=100, learning_rate=learning_rate)   # goes first for GPU allocation

I = class_images.ImageClass()
F = class_features.ResnetFeaturesClass()
R = class_surrogate_responses.SurrogateResponsesClass(pretrained_CNN=pretrained_CNN)

nums_ensemble_networks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nums_ensemble_networks = [1]

nums_ensemble_networks = [25]

num_runs = 5
for irun in [3]: #range(num_runs):
	seed_index = irun * 999

	# get test set
	seed(seed_index)
	set_random_seed(seed_index)
	features_test, responses_test = get_test_set(I,F,R)

	# run for each num networks
	for num_ensemble_networks in nums_ensemble_networks:
		del M
		seed(seed_index + num_ensemble_networks)
		set_random_seed(seed_index + num_ensemble_networks)

		M = class_model_ensemble.ModelEnsembleClass(num_models=num_ensemble_networks, num_output_vars=100, learning_rate=learning_rate, seed_index=seed_index)
		A = class_random_selection.RandomSelection()
		run_model(I, F, R, M, A, pretrained_CNN, 'random_selection', num_ensemble_networks, num_sessions, irun=irun)

