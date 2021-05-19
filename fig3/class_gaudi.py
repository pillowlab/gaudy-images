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



	# NOT CHANGED
	# changes values to nearest bin intensity value (e.g., [0,255], [0,128,255], ...)
	def get_images_to_show_nearest_bin_value(self, I, F, R, M, num_images=500, num_chosen=250, bins=[]):
		# I: image class
		# bins contain the values spread out over pixel intensities, e.g. [0,255], [0, 128, 255], etc.

		if bins.size == 0:
			bins = np.array([0, 255])

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			for ichannel in range(3):
				img_channel = np.copy(imgs[iimg,:,:,ichannel])

				diffs = np.abs(img_channel[:,:,np.newaxis] - np.ones((img_channel.shape[0], img_channel.shape[1], bins.size)) * bins)
				inds = np.argmin(diffs,axis=-1)

				img_channel = bins[inds]

				img_channel = np.clip(img_channel, a_min=0, a_max=255)

				imgs[iimg,:,:,ichannel] = img_channel

		return imgs


	def get_images_to_show_contrast_levels(self, I, F, R, M, contrast_level=1.0, num_images=500, num_chosen=250):
		# I: image class

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			img = imgs[iimg]

			c = np.quantile(img,0.05)

			img = img - c  # remove background lumination
			img = img * contrast_level # increase contrast

			img[img < 0] = 0
			img[img > 255] = 255

			imgs[iimg] = img

		return imgs


	# NOT CHANGED
	def get_images_to_show_limit_levels(self, I, F, R, M, fraction_contrast=1.0, num_images=500, num_chosen=250):
		# I: image class

		pixel_means = np.array([123.68, 116.78, 103.94])

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			for ichannel in range(3):
				img_channel = imgs[iimg,:,:,ichannel] - pixel_means[ichannel]
				img_channel[img_channel < 0] = -pixel_means[ichannel] * fraction_contrast
				img_channel[img_channel >= 0] = (255 - pixel_means[ichannel]) * fraction_contrast

				img_channel += pixel_means[ichannel]
				img_channel = np.clip(img_channel, a_min=0, a_max=255)

				imgs[iimg,:,:,ichannel] = img_channel

		return imgs


	def get_images_to_show_smoothing_before_gaudi(self, I, F, R, M, smoothing_sigma=1.0, num_images=500, num_chosen=250):
		# I: image class

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			img = imgs[iimg]

			# smoothe before gaudi transformation
			img = ndimage.filters.gaussian_filter(img, (smoothing_sigma, smoothing_sigma, 0.))

			m = np.mean(img)
			img[img < m] = 0
			img[img >= m] = 255
			imgs[iimg] = img

		return imgs


	def get_images_to_show_smoothing_after_gaudi(self, I, F, R, M, smoothing_sigma=1.0, num_images=500, num_chosen=250):
		# I: image class

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			img = imgs[iimg]
			m = np.mean(img)
			img[img < m] = 0
			img[img >= m] = 255

			# smoothe after gaudi transformation
			img = ndimage.filters.gaussian_filter(img, (smoothing_sigma, smoothing_sigma, 0.))

			imgs[iimg] = img

		return imgs


	# NOT CHANGED
	def get_images_to_show_nearmedian_values_rail_first(self, I, F, R, M, quantile=0.1, num_images=500, num_chosen=250):
		# I: image class

		pixel_means = np.array([123.68, 116.78, 103.94])

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			img = np.copy(imgs[iimg])

			# recenter image
			for ichannel in range(3):
				img[:,:,ichannel] -= pixel_means[ichannel]

			q = np.quantile(np.abs(img), quantile)

			img_gaudi = np.copy(img)
			for ichannel in range(3):
				img_channel = img_gaudi[:,:,ichannel]

				img_channel[img_channel < 0] = -500
				img_channel[img_channel >= 0] = 500

				img_gaudi[:,:,ichannel] = img_channel

			img[np.abs(img) < q] = img_gaudi[np.abs(img) < q]

			for ichannel in range(3):
				img[:,:,ichannel] += pixel_means[ichannel]

			img = np.clip(img, a_min=0, a_max=255)

			imgs[iimg] = img

		return imgs


	def get_images_to_show_near_edges(self, I, F, R, M, percent_changed_pixels=0.1, num_images=500, num_chosen=250):
		# I: image class

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):
			
			img = np.copy(imgs[iimg])

			img_2d = np.mean(img, axis=-1)
			img_2d = ndimage.gaussian_filter(img_2d, (1.0, 1.0))

			img1 = ndimage.sobel(np.copy(img_2d), axis=0)
			img2 = ndimage.sobel(np.copy(img_2d), axis=1)
			img_2d = np.hypot(img1, img2)

			img_2d /= np.max(img_2d)
			img_2d = img_2d * 255

			img_2d = np.clip(img_2d, a_min=0, a_max=255)

			m = np.quantile(img_2d, 1-percent_changed_pixels)

			img_2d[img_2d < m] = 0
			img_2d[img_2d >= m] = 1

			img_mask = np.copy(imgs[iimg])
			for ichannel in range(3):
				img_mask[:,:,ichannel] = img_2d

			img_gaudi = np.copy(imgs[iimg])
			m = np.mean(img_gaudi)
			img_gaudi[img_gaudi < m] = 0
			img_gaudi[img_gaudi >= m] = 255

			img_new = np.copy(imgs[iimg])

			img_new[img_mask == 1] = img_gaudi[img_mask == 1]

			imgs[iimg] = img_new

		return imgs


	def get_images_to_show_far_from_edges(self, I, F, R, M, percent_changed_pixels=0.1, num_images=500, num_chosen=250):
		# I: image class

		pixel_means = np.array([123.68, 116.78, 103.94])

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):
			
			img = np.copy(imgs[iimg])

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

			img_mask = np.copy(imgs[iimg])
			for ichannel in range(3):
				img_mask[:,:,ichannel] = img_2d

			img_gaudi = np.copy(imgs[iimg])
			m = np.mean(img_gaudi)
			img_gaudi[img_gaudi < m] = 0
			img_gaudi[img_gaudi >= m] = 255

			img_new = np.copy(imgs[iimg])

			img_new[img_mask == 0] = img_gaudi[img_mask == 0]

			imgs[iimg] = img_new

		return imgs


	# NOT CHANGED
	def get_images_to_show_farmedian_values_rail_first(self, I, F, R, M, quantile=0.1, num_images=500, num_chosen=250):
		# I: image class

		pixel_means = np.array([123.68, 116.78, 103.94])

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			img = np.copy(imgs[iimg])

			# recenter image
			for ichannel in range(3):
				img[:,:,ichannel] -= pixel_means[ichannel]

			q = np.quantile(np.abs(img), 1.0-quantile)

			img_gaudi = np.copy(img)
			for ichannel in range(3):
				img_channel = img_gaudi[:,:,ichannel]

				img_channel[img_channel < 0] = -500
				img_channel[img_channel >= 0] = 500

				img_gaudi[:,:,ichannel] = img_channel

			img[np.abs(img) >= q] = img_gaudi[np.abs(img) >= q]

			for ichannel in range(3):
				img[:,:,ichannel] += pixel_means[ichannel]

			img = np.clip(img, a_min=0, a_max=255)

			imgs[iimg] = img

		return imgs


	# NOT CHANGED
	def get_images_to_show_random_pixels(self, I, F, R, M, quantile=0.1, num_images=500, num_chosen=250):
		# I: image class

		pixel_means = np.array([123.68, 116.78, 103.94])

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		# gaudi 1/2 of the images
		for iimg in range(num_chosen):

			img = np.copy(imgs[iimg])

			# recenter image
			for ichannel in range(3):
				img[:,:,ichannel] -= pixel_means[ichannel]

			img_gaudi = np.copy(img)
			for ichannel in range(3):
				img_channel = img_gaudi[:,:,ichannel]

				img_channel[img_channel < 0] = -500
				img_channel[img_channel >= 0] = 500

				img_gaudi[:,:,ichannel] = img_channel

			if quantile > 0:
				mask = np.zeros((img.size,))
				inds = np.random.permutation(mask.size)
				inds = inds[:int(np.floor(inds.size * quantile))]
				mask[inds] = 1
				mask = np.reshape(mask, (img.shape))

				img[mask == 1] = img_gaudi[mask == 1]

			for ichannel in range(3):
				img[:,:,ichannel] += pixel_means[ichannel]

			img = np.clip(img, a_min=0, a_max=255)

			imgs[iimg] = img

		return imgs
		

	def get_images_to_show_change_colors(self, I, F, R, M, change_color_type='nochange', num_images=500, num_chosen=250):
		# I: image class

		imgs = I.get_random_natural_images(num_random_images=num_images)
			# returns imgs (*not* re-centered)

		for iimg in range(num_chosen):
			img = np.copy(imgs[iimg])

			if change_color_type == 'nochange':
				m = np.mean(img)
				img[img < m] = 0
				img[img >= m] = 255

			elif change_color_type == 'grayscale':
				# make image grayscale
				img_grayscale = np.mean(img,axis=-1)
				for ichannel in range(3):
					img[:,:,ichannel] = img_grayscale  

				# then make grayscale image gaudy
				m = np.mean(img)
				img[img < m] = 0
				img[img >= m] = 255

				img_mean = np.mean(img,axis=-1)
				for ichannel in range(3):
					img[:,:,ichannel] = img_mean

			elif change_color_type == 'blackwhiteonly':

				# identify rgb values that are all <0 or >0
				m = np.mean(img)
				inds_black = (img[:,:,0] < m) * (img[:,:,1] < m) * (img[:,:,2] < m)
				inds_white = (img[:,:,0] >= m) * (img[:,:,1] >= m) * (img[:,:,2] >= m)

				img[inds_black] = 0
				img[inds_white] = 255

			elif change_color_type == 'redonly':
				ichannel = 0
				m = np.mean(img)
				img_channel = img[:,:,ichannel]
				img_channel[img_channel < m] = 0
				img_channel[img_channel >= m] = 255

				img[:,:,ichannel] = img_channel

			elif change_color_type == 'greenonly':
				ichannel = 1
				m = np.mean(img)
				img_channel = img[:,:,ichannel]
				img_channel[img_channel < m] = 0
				img_channel[img_channel >= m] = 255

				img[:,:,ichannel] = img_channel

			elif change_color_type == 'blueonly':
				ichannel = 2
				m = np.mean(img)
				img_channel = img[:,:,ichannel]
				img_channel[img_channel < m] = 0
				img_channel[img_channel >= m] = 255

				img[:,:,ichannel] = img_channel
				
			elif change_color_type == 'inverted':
				m = np.mean(img)
				img_temp = np.copy(img)
				img_temp[img < m] = 255
				img_temp[img >= m] = 0
				img = img_temp

			elif change_color_type == 'belowonly':
				m = np.mean(img)
				img[img < m] = 0

			elif change_color_type == 'aboveonly':
				m = np.mean(img)
				img[img >= m] = 255

			imgs[iimg] = img

		return imgs



### NOTES:



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
