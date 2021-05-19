# class for random selection

import numpy as np
import scipy.ndimage as ndimage


class GaudiSelectionClass:

	# rails
	def get_images_to_show(self, I, F, R, M, num_images=500, num_chosen=250):
		# I: image class

		pixel_means = np.array([123.68, 116.78, 103.94])

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



### NOTES:


		# # gaudi 1/2 of the images
		# for iimg in range(num_chosen):

		# 	m = np.mean(imgs[iimg,:,:,:])
			
		# 	for ichannel in range(3):
		# 		img_channel = imgs[iimg,:,:,ichannel] - m
		# 		img_channel[img_channel < 0] = -500
		# 		img_channel[img_channel >= 0] = 500

		# 		img_channel += m
		# 		img_channel = np.clip(img_channel, a_min=0, a_max=255)

		# 		imgs[iimg,:,:,ichannel] = img_channel

		# return imgs


	# # scale each pixel
	# def get_images_to_show(self, I, F, R, M, num_images=500, num_chosen=250):
	# 	# I: image class

	# 	imgs = I.get_random_natural_images(num_random_images=num_images)

	# 	# bleach 1/2 of the images
	# 	for iimg in range(num_chosen):
	# 		for ichannel in range(3):
	# 			img_channel = imgs[iimg,:,:,ichannel]

	# 			m = np.mean(img_channel)
	# 			img_channel = (img_channel-m)*20. + m

	# 			imgs[iimg,:,:,ichannel] = img_channel

	# 		imgs[iimg] = I.clip_image_resnet(imgs[iimg])

	# 	return imgs


	# rails
	# def get_images_to_show(self, I, F, R, M, num_images=500, num_chosen=250):
	# 	# I: image class

	# 	imgs = I.get_random_natural_images(num_random_images=num_images)

	# 	# bleach 1/2 of the images
	# 	for iimg in range(num_chosen):
	# 		img = I.reverse_preprocessing(imgs[iimg])

	# 		img[img <= 128.] = 0.
	# 		img[img > 128.] = 255.


	# 		imgs[iimg] = I.image_to_resnet_format(img)
	# 	return imgs
