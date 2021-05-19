

import numpy as np

from scipy import ndimage
from skimage.filters import gabor_kernel


class GaborResponsesClass:

### INIT FUNCTION

	def __init__(self, passthrough_function='linear'):

		# create gabor filter
		kernel_small = gabor_kernel(frequency=0.1, bandwidth=0.5, theta=np.pi/4)
			# returns 49 x 49 gabor filter in the 45deg direction
		kernel = np.zeros((112,112))
		kernel[31:80,31:80] = kernel_small.real * 1e3  # scale to have max/min around 1/-1
		self.kernel = kernel

		self.passthrough_function = passthrough_function


	def get_responses_from_imgs(self, imgs):

		num_images = imgs.shape[0]
		responses = np.zeros((num_images,))
		for iimg in range(num_images):
			responses[iimg] = np.sum(self.kernel * imgs[iimg])

		if self.passthrough_function == 'relu':
			responses[responses < 0] = 0.
		elif self.passthrough_function == 'sigmoid':
			s = 1000.  # estimate of std for this kernel filter
			offset = 0.
			responses = 1. / (1. + np.exp(-(responses - offset)/s))

		return responses