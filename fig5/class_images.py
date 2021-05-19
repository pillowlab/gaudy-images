# image class to handle basic pre-processing and preparing for plotting

import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.preprocessing import image

import zipfile

class ImageClass:

	def __init__(self):
		self.resnet_pixel_means = np.array([103.94, 116.78, 123.68])  # take from Kzhivesky 2012 and https://forums.fast.ai/t/how-is-vgg16-mean-calculated/4577/10
						# pixel mean for training set of imagenet, in bgr format

		# image folder path
		self.image_path = '/jukebox/pillow/bcowley/adroit/data/image_dataset/images_zipped/'

			# storing images as: image_path + '12345/12345-1999.jpg' as an example, 0 to 1999 in each folder

		self.image_folder_ranges = [0, 6500]
		self.num_images_per_folder = 2000


	def clip_image_resnet(self, image_resnet):
		# clips the new image to be within range of pixel intensity space (between 0 and 255, but needs to be recentered
		#		for resnet images)

		upper_bounds = 255. - self.resnet_pixel_means
		lower_bounds = 0. - self.resnet_pixel_means

		for ichannel in range(3):
			image_resnet[:,:,:,ichannel] = np.clip(image_resnet[:,:,:,ichannel], lower_bounds[ichannel], upper_bounds[ichannel])

		return image_resnet


	def check_image_resnet_in_bounds(self, image_resnet):
		# clips the new image to be within range of pixel intensity space (between 0 and 255, but needs to be recentered
		#		for resnet images)

		upper_bounds = 255. - self.resnet_pixel_means + 5
		lower_bounds = 0. - self.resnet_pixel_means - 5

		for ichannel in range(3):
			if np.any(np.bitwise_or(image_resnet[:,:,:,ichannel] < lower_bounds[ichannel], image_resnet[:,:,:,ichannel] > upper_bounds[ichannel]) == 1):
				return False

		return True


	def reverse_preprocessing(self, imgs_resnet):
		# given img in resnet form, convert back to original rgb form

		imgs = []
		for iimg in range(imgs_resnet.shape[0]):
			img = imgs_resnet[iimg] + self.resnet_pixel_means

			img = np.clip(img, 0, 255).astype('uint8')

			img = img[:,:,::-1]  # so Resnet uses BGR instead of RGB lol, so reverse that as well
			
			imgs.append(img)

		return np.asarray(imgs)


	def get_white_noise_image(self, num_images=1):
		# get white noise image

		imgs = []
		for iimg in range(num_images):
			img = np.random.uniform(size=(224,224,3)) * 50 + 128  # guaranteed to be in bounds
			imgs.append(img)

		return np.asarray(imgs)


	def get_random_natural_images(self, num_random_images=1, target_size=(224, 224)):
		# returns random natural images (num_images x num_pixels x num_pixels x 3)
		# 
		# NOTE: if using to train neural net, you need to transform to resnet format!

		imgs = []

		for iimg in range(num_random_images):

			# choose folder randomly
			folder_index = np.random.randint(low=self.image_folder_ranges[0], high=self.image_folder_ranges[1])
			img_index = np.random.randint(self.num_images_per_folder)

			archive = zipfile.ZipFile(self.image_path + '{:05d}.zip'.format(folder_index), 'r')

			img_tag = '{:05d}-{:04}.jpg'.format(folder_index, img_index)

			img = image.load_img(archive.open(img_tag), target_size=target_size)
			img = image.img_to_array(img)

			imgs.append(img)

		return np.asarray(imgs)
		

	def get_images_from_zipfile(self, zipfileandpath, inds_image):
		# returns raw images

		imgs = []

		archive = zipfile.ZipFile(zipfileandpath, 'r')

		names = archive.namelist()

		for ind_image in inds_image:

			
			img_tag = 'image{:04}.jpg'.format(ind_image)

			for name in names: # could be a top folder, so this handles that
				if img_tag in name:
					
					img = image.load_img(archive.open(name), target_size=(224, 224))
					img = image.img_to_array(img)

					imgs.append(img)
					break

		return np.asarray(imgs)


	def transform_images_to_resnet_format(self, imgs_raw):
		# transforms normal images (0 to 255 pixel intensities)
		#	to resnet format (recentered and bgr)
		#	imgs: (num_images x num_pixels x num_pixels x 3), rgb images
		#	

		imgs = np.copy(imgs_raw)

		num_images = imgs.shape[0]

		for iimg in range(num_images):
			imgs[iimg,:,:,:] = preprocess_input_resnet50(imgs[iimg,:,:,:])

		return imgs


