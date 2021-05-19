# change the contrast levels
#  high-contrast: [0,255]
#  low-contrast: [115, 142]


import numpy as np

import class_model
import class_images
import class_features
import class_surrogate_responses

import class_random_selection
import class_gaudi

from numpy.random import seed
from tensorflow import set_random_seed

import scipy.ndimage as ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from skimage import feature


### HELPER FUNCTIONS

def get_test_set(I, F, R):

	imgs = I.get_random_natural_images(num_random_images=4000)

	features_test = F.get_features_from_imgs(imgs)
	responses_test = R.get_responses_from_imgs(imgs)

	return features_test, responses_test

def run_model(I, F, R, M, A, pretrained_CNN, save_filetag, num_sessions, irun=0, percent_changed_pixels=0.1):

	corrs_test = []

	print('preparing to run ' + save_filetag + '...')

	for isession in range(num_sessions):

		imgs_train = A.get_images_to_show_far_from_edges(I, F, R, M, percent_changed_pixels=percent_changed_pixels)

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

		print('{:s}: corr_test = {:0.4f}'.format(save_filetag, corr_test))

	np.save('./results/far_from_edges/corrs_test_' + pretrained_CNN + '_' + str(percent_changed_pixels*100) + 'percent_changed_pixels_run{:d}.npy'.format(irun), np.asarray(corrs_test))


def transform_image_to_far_from_edges(img_raw, percent_changed_pixels=0.1):
	# I: image class

	img = np.copy(img_raw)

	img_2d = np.mean(img, axis=-1)
	img_2d = ndimage.gaussian_filter(img_2d, (1.0, 1.0))

	img1 = ndimage.sobel(np.copy(img_2d), axis=0)
	img2 = ndimage.sobel(np.copy(img_2d), axis=1)
	img_2d = np.hypot(img1, img2)

	img_2d /= np.max(img_2d)
	img_2d = img_2d * 255

	img_2d = np.clip(img_2d, a_min=0, a_max=255)

	m = np.quantile(img_2d, percent_changed_pixels)

	img_2d[img_2d < m] = 0
	img_2d[img_2d >= m] = 1

	img_mask = np.copy(img_raw)
	for ichannel in range(3):
		img_mask[:,:,ichannel] = img_2d

	img_gaudi = np.copy(img_raw)
	m = np.mean(img_gaudi)
	img_gaudi[img_gaudi < m] = 0
	img_gaudi[img_gaudi >= m] = 255

	img_new = np.copy(img_raw)

	img_new[img_mask == 0] = img_gaudi[img_mask == 0]

	print(np.sum(img_mask == 1)/img_mask.size)

	img = img_new

	return img






### MAIN SCRIPT

pretrained_CNN = 'VGG19'  

num_sessions = 30
num_epochs = 5
num_models = 1

seed_index = 100;

# get models set up
seed(seed_index)
set_random_seed(seed_index)

learning_rate = 1e-1
	
M = class_model.ModelEnsembleClass(num_models=num_models, num_output_vars=100, learning_rate=learning_rate)   # goes first for GPU allocation

I = class_images.ImageClass()
F = class_features.ResnetFeaturesClass()
R = class_surrogate_responses.SurrogateResponsesClass(pretrained_CNN=pretrained_CNN)


if False:
	print('getting test set...')
	features_test, responses_test = get_test_set(I,F,R)

	percents_changed_pixels = [0., 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

# train models with different percent changed pixels far from edge

	for irun in range(5):
		seed_index = 100 * irun
		seed(seed_index)
		set_random_seed(seed_index)

		del M
		M = class_model.ModelEnsembleClass(num_models=num_models, num_output_vars=100, learning_rate=learning_rate, seed_index=seed_index)   # goes first for GPU allocation
		M.save_models('initial_training_{:s}'.format(pretrained_CNN))
		
		A = class_gaudi.GaudiSelectionClass()

		for percent_changed_pixels in percents_changed_pixels:
			print('running {:f} percent changed pixels'.format(percent_changed_pixels*100))
			M.load_models('initial_training_{:s}'.format(pretrained_CNN))  # reload initial model each time
			run_model(I, F, R, M, A, pretrained_CNN, 'gaudi', num_sessions, irun=irun, percent_changed_pixels=percent_changed_pixels)


### plot images with different contrast levels
if True:
	percents_changed_pixels = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	f = plt.figure(figsize=(10,10))

	img_orig = I.get_example_image()

	ipanel = 1

	plt.subplot(4,4,ipanel)
	plt.imshow(img_orig.astype('uint8'), vmin=0, vmax=255)
	plt.axis('off')
	plt.title('orig')
	ipanel += 1

	for percent_changed_pixels in percents_changed_pixels:
		
		img = transform_image_to_far_from_edges(img_orig, percent_changed_pixels=percent_changed_pixels)

		plt.subplot(4,4,ipanel)
		plt.imshow(img.astype('uint8'), vmin=0, vmax=255)
		plt.axis('off')
		plt.title('{:d}'.format(int(percent_changed_pixels*100)))
		ipanel += 1

	f.savefig('./figs/thumbnails_far_from_edges.pdf')


