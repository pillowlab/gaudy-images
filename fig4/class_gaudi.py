# class for random selection

import numpy as np
import scipy.ndimage as ndimage


class GaudiSelectionClass:

	# rails
	def get_images_to_show(self, I, F, R, M, num_images=500, num_chosen=250):
		# I: image class

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			img = imgs[iimg]

			m = np.mean(img)
			img[img < m] = 0
			img[img >= m] = 255

			imgs[iimg] = img

		return imgs
