

import numpy as np

import class_model
import class_images
import class_features
import class_surrogate_responses

import class_random_selection
import class_gaudi

from numpy.random import seed
from tensorflow import set_random_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



### MAIN SCRIPT

pretrained_CNN = 'VGG19'
learning_rate = 1e-1

# M = class_model.ModelEnsembleClass(num_models=1, num_output_vars=100, learning_rate=learning_rate)   # goes first for GPU allocation

I = class_images.ImageClass()
F = class_features.ResnetFeaturesClass()
R = class_surrogate_responses.SurrogateResponsesClass(pretrained_CNN=pretrained_CNN)

# get 500 images and responses each for normal and gaudy images
if True:
	seed(31490)

	imgs_random = I.get_random_natural_images(num_random_images=500)
	# img_sky = I.get_example_image()
	# imgs_random = np.concatenate((imgs_random, img_sky[np.newaxis,:,:,:]), axis=0)

	responses_random = R.get_responses_from_imgs(imgs_random)

	imgs_gaudy = np.copy(imgs_random)
	imgs_gaudy = I.get_gaudy_images(imgs_gaudy)

	responses_gaudy = R.get_responses_from_imgs(imgs_gaudy)

	print(imgs_random.shape)
	print(imgs_gaudy.shape)
	
	# save images and responses (for Adv Python 2020 class)
	np.save('./results/imgs_normal.npy', imgs_random)
	np.save('./results/imgs_gaudy.npy', imgs_gaudy)
	np.save('./results/responses_normal.npy', responses_random)
	np.save('./results/responses_gaudy.npy', responses_gaudy)

# apply PCA to responses (combined)
if True:
	responses_combined = np.concatenate((responses_random, responses_gaudy), axis=1)

	print(responses_combined.shape)
	pca = PCA(n_components=2)
	pca.fit(responses_combined.T)

	Z_rand = pca.transform(responses_random.T).T
	Z_gaudi = pca.transform(responses_gaudy.T).T

	f = plt.figure()

	plt.plot(Z_rand[0,:], -Z_rand[1,:], '.k', alpha=0.5, markeredgewidth=0)
	plt.plot(Z_gaudi[0,:], -Z_gaudi[1,:], '.r', alpha=0.5, markeredgewidth=0)

	# identify two dots (one rand, one gaudi) to point out, Sky Rafidi example

	plt.plot(Z_rand[0,-1], -Z_rand[1,-1], '.b')
	plt.plot(Z_gaudi[0,-1], -Z_gaudi[1,-1], '.b')

	f.savefig('./figs/pca_top_2PCs_rand_vs_gaudy.pdf', transparent=True)

	f = plt.figure()

	plt.subplot(3,3,1)

	plt.imshow(imgs_random[-1].astype('uint8'), vmin=0, vmax=255)

	plt.subplot(3,3,2)
	plt.imshow(imgs_gaudy[-1].astype('uint8'), vmin=0, vmax=255)

	f.savefig('./figs/pca_top_2PCs_images.pdf', transparent=True)


# apply PCA separately to gaudi and random responses
if False:
	seed(31490)

	imgs_random = I.get_random_natural_images(num_random_images=5000)

	responses_random = R.get_responses_from_imgs(imgs_random)

	imgs_gaudy = np.copy(imgs_random)
	imgs_gaudy = I.get_gaudi_images(imgs_gaudy)

	responses_gaudy = R.get_responses_from_imgs(imgs_gaudy)

	pca = PCA(n_components=50)

	f = plt.figure()

	pca.fit(responses_random.T)
	plt.plot(pca.explained_variance_, 'k', label='random')

	pca.fit(responses_gaudy.T)
	plt.plot(pca.explained_variance_, 'r', label='gaudi')

	plt.xlabel('PC index')
	plt.ylabel('var')
	plt.legend()

	f.savefig('./figs/pca_explained_variance.pdf', transparent=True)



















