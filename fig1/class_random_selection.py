
import numpy as np


class RandomSelectionClass:

	def get_images_to_show(self, I, num_images=500):
		# I: image class

		return I.get_random_natural_images(num_random_images=num_images)