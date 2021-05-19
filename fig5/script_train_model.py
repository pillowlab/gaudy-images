

import numpy as np 

import class_model
import class_features
import class_data
import class_images

from tensorflow.keras import backend as K

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### MAIN SCRIPT

image_type = sys.argv[2]  # {'normal', 'gaudylike'}

M = class_model.ModelClass(num_models=1, learning_rate=5e-2)
if False:
	M.save_models(filetag='initial_model', save_folder='./results/saved_models/')
else:
	M.load_models(filetag='initial_model', load_folder='./results/saved_models/')

D = class_data.DataClass()
F = class_features.ResnetFeaturesClass()
I = class_images.ImageClass()

day_ids = [190924, 190925, 190926, 190927, 190928, 190929]

num_training_days = len(day_ids)

# get test data
if True:
	heldout_day_id = 190923
	imgs_test, responses1_test, responses2_test = D.get_images_and_split_responses(heldout_day_id, I)
	features_test = F.get_features_from_imgs(imgs_test)

	if False:
		m = np.mean(responses1_test,axis=1)[:,np.newaxis]
		s = np.std(responses1_test,axis=1)[:,np.newaxis]
		responses1_test = (responses1_test - m) / s   # z-scoring

		m = np.mean(responses2_test,axis=1)[:,np.newaxis]
		s = np.std(responses2_test,axis=1)[:,np.newaxis]
		responses2_test = (responses2_test - m) / s   # z-scoring

# train model 
#	train a little bit each day, up to 6 days
if True:

	np.random.seed(22222)

	num_epochs = 2

	frac_vars = []

	for ipass in range(5):

		r = np.random.permutation(num_training_days)

		for iday in range(num_training_days):

			day_id = day_ids[r[iday]]

			# get pre-sorted imgs/responses based on gaudiness
			if image_type == 'gaudylike':
				imgs_train = np.load('./results/images/imgs_gaudylike_day{:d}.npy'.format(day_id))
				responses_train = np.load('./results/responses/responses_gaudylike_day{:d}.npy'.format(day_id))
			elif image_type == 'normal':
				imgs_train = np.load('./results/images/imgs_normal_day{:d}.npy'.format(day_id))
				responses_train = np.load('./results/responses/responses_normal_day{:d}.npy'.format(day_id))

			features_train = F.get_features_from_imgs(imgs_train)
			# z-score responses_train such that each neuron provides a similar gradient
				# [is this needed?]
			if False:
				m = np.mean(responses_train,axis=1)[:,np.newaxis]
				s = np.std(responses_train,axis=1)[:,np.newaxis]
				responses_train = (responses_train - m) / s   # z-scoring

			for iepoch in range(num_epochs):
				M.train_models(features_train, responses_train, iepoch=iepoch)

		fv = np.mean(M.compute_test_frac_exp_var(features_test, responses1_test, responses2_test))

		frac_vars.append(fv)

		print('pass {:d}, fv = {:f}'.format(ipass, fv))

	np.save('./results/frac_vars_over_days.npy', np.asarray(frac_vars))




