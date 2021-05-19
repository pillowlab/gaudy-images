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

		self.image_folder_ranges = [0, 6000]

		self.random_images = np.array([])  # small store of random images to choose some
		self.index_random_images = 0  # keeps track of next random image to be chosen
			

	def clip_image(self, image_raw):
		# clips the new image to be within range of pixel intensity space (between 0 and 255, but needs to be recentered
		#		for resnet images)

		image = np.clip(image_raw, a_min=0, a_max=255)

		return image
		

	def clip_image_resnet(self, image_resnet):
		# clips the new image to be within range of pixel intensity space (between 0 and 255, but needs to be recentered
		#		for resnet images)

		image_4dims = False
		if len(image_resnet.shape) == 4:
			image_resnet = np.squeeze(image_resnet)
			image_4dims = True

		upper_bounds = 255. - self.resnet_pixel_means
		lower_bounds = 0. - self.resnet_pixel_means

		for ichannel in range(3):
			image_resnet[:,:,ichannel] = np.clip(image_resnet[:,:,ichannel], lower_bounds[ichannel], upper_bounds[ichannel])

		if image_4dims == True:
			image_resnet = image_resnet[np.newaxis,:,:,:]

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
		

	def image_to_resnet_format(self, img):
		img = preprocess_input_resnet50(img)

		return img
		

	def reverse_preprocessing(self, img):
		# given img in resnet form, convert back to original rgb form

		img = img + self.resnet_pixel_means

		img = np.clip(img, 0, 255).astype('uint8')

		img = img[:,:,::-1]  # so Resnet uses BGR instead of RGB lol, so reverse that as well
		return img


	def get_gaudi_image(self, img):
		# imgs in raw format, array (num_imgs, num_x, num_y, 3)

		m = np.mean(img)
		img[img < m] = 0
		img[img >= m] = 255

		return img


	def get_gaudi_images(self, imgs):
		# imgs in raw format, array (num_imgs, num_x, num_y, 3)
		
		for iimg in range(imgs.shape[0]):

			img = imgs[iimg]

			m = np.mean(img)
			img[img < m] = 0
			img[img >= m] = 255

			imgs[iimg] = img

		return imgs


	def get_2k_random_images(self):
		# get images from zipped files (2k images per file)
		# returns images in raw format (no re-centering)

		zip_index = np.random.randint(low=self.image_folder_ranges[0], high=self.image_folder_ranges[1])

		archive = zipfile.ZipFile(self.image_path + '{:05d}.zip'.format(zip_index), 'r')

		imgs = []
		for iimg in range(2000):
			img_tag = '{:05d}-{:04}.jpg'.format(zip_index, iimg)

			img = image.load_img(archive.open(img_tag), target_size=(224, 224))
			img = image.img_to_array(img)

			imgs.append(img)
		imgs = np.asarray(imgs)

		return imgs


	def get_2k_images_from_folder_index(self, folder_index):
		# folder index between 0 and 6000

		archive = zipfile.ZipFile(self.image_path + '{:05d}.zip'.format(folder_index), 'r')

		imgs = []
		for iimg in range(2000):
			img_tag = '{:05d}-{:04}.jpg'.format(folder_index, iimg)

			img = image.load_img(archive.open(img_tag), target_size=(224, 224))
			img = image.img_to_array(img)

			imgs.append(img)
		imgs = np.asarray(imgs)

		return imgs


	def get_image_from_folder_index(self, folder_index, image_index):
		# folder index between 0 and 6000
		# used for coreset AL

		archive = zipfile.ZipFile(self.image_path + '{:05d}.zip'.format(folder_index), 'r')

		img_tag = '{:05d}-{:04}.jpg'.format(folder_index, image_index)

		img = image.load_img(archive.open(img_tag), target_size=(224, 224))
		img = image.img_to_array(img)

		return img


	def get_white_noise_image(self):
		# get white noise image

		img = np.random.uniform(size=(1,224,224,3)) * 50 + 128

		for ichannel in range(3):
			img[:,:,:,ichannel] = img[:,:,:,ichannel] - self.resnet_pixel_means[ichannel]

		return img


	def get_random_natural_images(self, num_random_images=1):
		# get random natural image

		imgs = []
		for iimg in range(num_random_images):
			if self.random_images.size == 0 or self.index_random_images == 2000:
				self.random_images = self.get_2k_random_images()
				self.index_random_images = 0
			
			imgs.append(self.random_images[self.index_random_images,:,:,:])

			self.index_random_images += 1

		return np.asarray(imgs)


	def get_sky_rafidi_example(self):
		# get random natural image

		img_filepath = './results/sky_rafidi_example.jpg'

		img = image.load_img(img_filepath, target_size=(224, 224))
		img = image.img_to_array(img)

		return img


