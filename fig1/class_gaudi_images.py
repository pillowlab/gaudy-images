
import numpy as np


class GaudiImageClass:

	def get_images_to_show(self, I, num_images=500):
		# I: image class

		imgs = I.get_random_natural_images(num_random_images=num_images)

		for iimg in range(np.round(num_images).astype(int)):
			img = imgs[iimg]

			m = np.mean(img)
			img[img < m] = 0
			img[img >= m] = 255

			imgs[iimg] = img

		return imgs