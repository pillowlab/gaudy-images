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
import class_synthetic_gaudi
import class_synthetic_normal
import class_coreset_normal
import class_coreset_gaudy

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


def run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions, seed_index=50):

	corrs_test = []

	print('preparing to run ' + AL_algo + '...')

	for isession in range(num_sessions):

		imgs_train = A.get_images_to_show(I, F, R, M)

		if True:
			f = plt.figure(figsize=(15,15))
			for ipanel in range(16):
				plt.subplot(4,4,ipanel+1)
				plt.imshow(imgs_train[ipanel].astype('uint8'), vmin=0, vmax=255)
				plt.axis('off')

			f.savefig('./figs/{:s}/session{:d}.pdf'.format(AL_algo, isession))

			plt.close(f)
			
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

		np.save('./results/corrs_test_' + pretrained_CNN + '_' + AL_algo + '.npy', np.asarray(corrs_test))

		if AL_algo == 'random_selection':
			M.save_models('model_random_selection_{:s}'.format(pretrained_CNN))



### MAIN SCRIPT

# python -u script_test_AL_algorithms.py #gpu_id $AL_algo

pretrained_CNN = 'VGG19'  

AL_algo = sys.argv[2]

num_sessions = 50
num_epochs = 5
num_networks = 25

seed_index = 100;

# get models set up
seed(seed_index)
set_random_seed(seed_index)

learning_rate = 1e-1
	
M = class_model_ensemble.ModelEnsembleClass(num_models=num_networks, num_output_vars=100, learning_rate=learning_rate)   # goes first for GPU allocation

I = class_images.ImageClass()
F = class_features.ResnetFeaturesClass()
R = class_surrogate_responses.SurrogateResponsesClass(pretrained_CNN=pretrained_CNN)

print('getting test set...')
features_test, responses_test = get_test_set(I,F,R)

# save initial model
if True:
	features_train, responses_train = get_initial_train_set(I, F, R)
	M.train_models(features_train, responses_train)
	M.save_models('initial_training_{:s}_{:d}networks'.format(pretrained_CNN, num_networks))
	print('initial model saved')

if AL_algo == 'random_selection':
	A = class_random_selection.RandomSelection()
	M.load_models('initial_training_{:s}'.format(pretrained_CNN))  # reload initial model each time
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)

elif AL_algo == 'gaudi':
	A = class_gaudi.GaudiSelectionClass()
	M.load_models('initial_training_{:s}'.format(pretrained_CNN))  
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)

elif AL_algo == 'largebank_gaudi':
	A = class_largebank_gaudi_only.LargebankGaudiOnlyClass()
	M.load_models('initial_training_{:s}'.format(pretrained_CNN))  
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)

elif AL_algo == 'largebank_normal':
	A = class_largebank_normal_only.LargebankNormalOnlyClass()
	M.load_models('initial_training_{:s}'.format(pretrained_CNN))  
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)

elif AL_algo == 'largebank_normalandgaudi':
	A = class_largebank_gaudi_and_normal.LargebankGaudiNormalClass()
	M.load_models('initial_training_{:s}'.format(pretrained_CNN))  
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)

elif AL_algo == 'synthetic_gaudi':
	A = class_synthetic_gaudi.SyntheticGaudiClass(I, F, R, M)
	M.load_models('initial_training_{:s}'.format(pretrained_CNN))  
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)

elif AL_algo == 'synthetic_normal':
	A = class_synthetic_normal.SyntheticNormalClass(I, F, R, M)
	M.load_models('initial_training_{:s}'.format(pretrained_CNN))  
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)

elif AL_algo == 'coreset_normal':
	A = class_coreset_normal.CoresetNormalClass()
	M.load_models('initial_training_{:s}_{:d}networks'.format(pretrained_CNN, num_networks))  
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)

elif AL_algo == 'coreset_gaudy':
	A = class_coreset_gaudy.CoresetNormalClass()
	M.load_models('initial_training_{:s}_{:d}networks'.format(pretrained_CNN, num_networks))  
	run_model(I, F, R, M, A, pretrained_CNN, AL_algo, num_sessions)



	

