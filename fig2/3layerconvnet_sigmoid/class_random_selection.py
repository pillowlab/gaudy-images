# class for random selection

import numpy as np

class RandomSelection:

	def get_images_to_show(self, I, F, R, M, num_images=500):
		# I: image class

		return I.get_random_natural_images(num_random_images=num_images)





