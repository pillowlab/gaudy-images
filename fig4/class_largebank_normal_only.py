# class for pool-based selection
#
#	 from large pool of candidate images, chooses
#	 a small number that maximizes ensemble variance
#
#	- ensemble variance is measured as the median distance between model responses
#	- after finding the 2k images with the largest ensemble variance, we then
#		perform a coreset on them to take 250 images that cover the 2k set

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




class LargebankNormalOnlyClass:


### HELPER INTERNAL FUNCTIONS

	def compute_ens_vars(self, imgs_cand, F, M):
		# compute responses
			num_images = imgs_cand.shape[0]
			features_cand = F.get_features_from_imgs(imgs_cand)

			responses = np.zeros((M.num_models, M.num_output_vars, num_images))
			for imodel in range(M.num_models):
				responses[imodel,:,:] = M.get_predicted_responses_from_ith_model(features_cand, imodel)

		# estimating ens var as the median distance between models
			med_dists = []
			for iimg in range(num_images):
				m = np.median(distance.pdist(X=responses[:,:,iimg]))
				med_dists.append(m)

			ens_vars = np.array(med_dists)

			return ens_vars


	def get_max_ens_var_from_2k_block(self, imgs_chosen, ens_vars_chosen, I, F, M):
		# returns imgs_chosen, ens_vars_chosen
			num_images_per_block = 2000

		# get ensemble variances of current block
			imgs_cand = I.get_random_natural_images(num_images_per_block)
			ens_vars = self.compute_ens_vars(imgs_cand, F, M)

		# replace imgs in imgs_chosen with imgs in imgs_cand that have larger ensemble variance
			ens_vars_cand = np.concatenate((ens_vars, ens_vars_chosen), axis=0)
			inds_sorted = np.argsort(ens_vars_cand)

			imgs = np.concatenate((imgs_cand, imgs_chosen), axis=0)

			return imgs[inds_sorted[-num_images_per_block:]], ens_vars_cand[inds_sorted[-num_images_per_block:]]


	def get_coreset_images(self, F, M, imgs_cand):
		# choose images that represent the coreset of their responses
		#	250 images from 2k

		features = F.get_features_from_imgs(imgs_cand)
		features = np.reshape(features, (features.shape[0],-1)).T

		inds_coreset = coreset(X=features, num_coreset_samples=250)

		return imgs_cand[inds_coreset]


	def get_ensemble_variance_imgs(self, I, F, R, M):
		# full function
		#	checks through the entire pool of candidate images to find the ones that maximize ensemble variance

		# get initial set of 2k images + ens vars
			imgs_chosen = I.get_random_natural_images(2000)
			ens_vars_chosen = self.compute_ens_vars(imgs_chosen, F, M)

		# keep checking if images have larger ens var than imgs_chosen
			for iblock in range(0,self.num_cand_images,2000):
				print('   block {:d}'.format(iblock))
				imgs_chosen, ens_vars_chosen = self.get_max_ens_var_from_2k_block(imgs_chosen, ens_vars_chosen, I, F, M)

		# choose maximally diverse images (250 out of the 2k)
			imgs_chosen = self.get_coreset_images(F, M, imgs_chosen)

			return imgs_chosen



### INIT FUNCTION

	def __init__(self, num_rands=250, pool_size=80000):

		self.num_rands = num_rands
		self.num_chosen = 500 - self.num_rands
		self.num_cand_images = pool_size

		

### EXTERNAL CALL FUNCTIONS

	def get_images_to_show(self, I, F, R, M):

		imgs_chosen = self.get_ensemble_variance_imgs(I, F, R, M)

		if self.num_rands > 0:
			imgs_rand = I.get_random_natural_images(self.num_rands)

		if self.num_rands == 500:
			imgs_chosen = imgs_rand
		elif self.num_rands > 0:
			imgs_chosen = np.concatenate((imgs_chosen, imgs_rand), axis=0)

		return imgs_chosen



