# class for coreset normal
#
#	 from large pool of candidate images, chooses
#	 the coreset between past and currently chosen images
#	 (uses predicted responses)
#


import sys
sys.path.append('/usr/people/bcowley/adroit/experiment/cosyne/helper_functions')
from coreset import coreset

from tensorflow.keras import backend as K

from tensorflow.keras import losses as losses

from tensorflow.keras.models import Model

import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from numpy import linalg as LA
from sklearn.metrics import pairwise_distances
import scipy.spatial.distance as distance

import time




class CoresetNormalClass:


### HELPER INTERNAL FUNCTIONS

	def get_responses(self, F, M, imgs):
		features = F.get_features_from_imgs(imgs)
		responses = M.get_predicted_ensemble_responses(features)

		return responses


	def get_min_dists(self, responses_prev, responses_cand_block, min_dists_block, F, M):
		# min_dists: (2k,), min dists for these cand_images

		# compute dists
		dists = distance.cdist(responses_cand_block.T, responses_prev.T)  # (num_cand, num_prev)
		dists = np.concatenate((dists, min_dists_block[:,np.newaxis]), axis=1)

		return np.min(dists, axis=1)


	def get_cand_image_with_max_min_dist(self, I, cand_folder_indices, responses, min_dists):
		# find image with largest min dist, add to chosen images
		#	min_dists: (list: num_cand_folders), list of min_dist blocks, where min_dists[ifolder] is (2000,) min dists for that folder's images
		#	responses: (list: num_cand_folders), list of responses, responses[ifolder] is (100,2000)
		ifolder_max = -1
		iimage_max = -1
		maxmin_dist = 0.
		num_cand_folders = len(min_dists)
		for ifolder in range(num_cand_folders):
			iimage = np.argmax(min_dists[ifolder])
			if min_dists[ifolder][iimage] > maxmin_dist:
				maxmin_dist = min_dists[ifolder][iimage]
				ifolder_max = ifolder
				iimage_max = iimage

		print(maxmin_dist)
		next_chosen_image = I.get_image_from_folder_index(folder_index=cand_folder_indices[ifolder_max], image_index=iimage_max)

		return next_chosen_image, responses[ifolder_max][:,iimage_max][:,np.newaxis]


	def get_coreset_imgs(self, I, F, R, M):
		# full function

		chosen_images = []

		# first choose folders for candidate images
		num_cand_folders = np.floor(self.num_cand_images/2000).astype('int')  # each folder has 2k images
		cand_folder_indices = np.random.permutation(6000)
		cand_folder_indices = cand_folder_indices[:num_cand_folders]

		# next, get responses for each candidate image and previous images
		if self.coreset_images.size > 0:
			responses_prev = self.get_responses(F, M, previous_images)

		responses_cand = []
		for ifolder in range(num_cand_folders):
			cand_images = I.get_2k_images_from_folder_index(folder_index=cand_folder_indices[ifolder])
			responses_cand.append(self.get_responses(F, M, cand_images))

		print('all responses are computed')

		# first pass: for each candidate image, compute min distance between previously-shown images
		#   choose image with largest dist, add to "chosen images"
		#	keep running tab of max min images
		if self.coreset_images.size > 0:  # ignore if no previously-shown images
			min_dists = []
			for ifolder in range(num_cand_folders):

				min_dists_initial = 1e7 * np.ones((2000,)) # assume all images are far away initially
				min_dists.append(self.get_min_dists(responses_prev, responses_cand[ifolder], min_dists_initial, F, M))

			next_chosen_image, responses_prev = self.get_cand_image_with_max_min_dist(I, cand_folder_indices, responses_cand, min_dists)
			chosen_images.append(next_chosen_image)
				# NOTE: no need to update min dists, we will recompute that distance in a later pass
		else:
			folder_index = cand_folder_indices[0]
			cand_image = I.get_image_from_folder_index(folder_index=folder_index, image_index=0)
			chosen_images.append(cand_image) 
			responses_prev = self.get_responses(F, M, cand_images[0][np.newaxis,:,:,:])

			min_dists = [] # initialize to large values (since no previously-shown images)
			min_dists_initial_block = 1e7 * np.ones((2000,))
			for ifolder in range(num_cand_folders):
				min_dists.append(min_dists_initial_block)

		# second pass: for each candidate image, compute min distance between chosen images and previously-shown images
		#	checks through the entire pool of candidate images to find the ones that maximize the minimum distance
		while len(chosen_images) < self.num_chosen:

			print(len(chosen_images))
			# update min dists with current chosen image
			for ifolder in range(num_cand_folders):
				min_dists[ifolder] = self.get_min_dists(responses_prev, responses_cand[ifolder], min_dists[ifolder], F, M)

			next_chosen_image, responses_prev = self.get_cand_image_with_max_min_dist(I, cand_folder_indices, responses_cand, min_dists)
			chosen_images.append(next_chosen_image)

		return np.array(chosen_images)



### INIT FUNCTION

	def __init__(self, num_rands=250, pool_size=80000):

		self.num_rands = num_rands
		self.num_chosen = 500 - self.num_rands
		self.num_cand_images = pool_size

		self.coreset_images = np.array([])
		

### EXTERNAL CALL FUNCTIONS

	def get_images_to_show(self, I, F, R, M):

		imgs_chosen = self.get_coreset_imgs(I, F, R, M)

		if self.num_rands > 0:
			imgs_rand = I.get_random_natural_images(self.num_rands)

		if self.num_rands == 500:
			imgs_chosen = imgs_rand
		elif self.num_rands > 0:
			imgs_chosen = np.concatenate((imgs_chosen, imgs_rand), axis=0)

			# add chosen chosen images to previously chosen coreset images
			if self.coreset_images.size == 0:
				self.corese_images = imgs_chosen
			else:
				self.coreset_images = np.concatenate((self.coreset_images, imgs_chosen), axis=0)
					
		return imgs_chosen



