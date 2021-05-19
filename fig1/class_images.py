# image class to handle basic pre-processing and preparing for plotting

import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.preprocessing import image

import zipfile

class ImageClass:

	def __init__(self):
		
		# image folder path
		self.image_path = '/jukebox/pillow/bcowley/adroit/data/image_dataset/images_zipped/'

			# storing images as: image_path + '12345/12345-1999.jpg' as an example, 0 to 1999 in each folder

		self.image_folder_ranges = [0, 6000]

		self.random_images = np.array([])  # small store of random images to choose some
		self.index_random_images = 0  # keeps track of next random image to be chosen


	def get_2k_random_images(self):
		# get images from zipped files (2k images per file)
		# returns images in raw format [0,256]

		zip_index = np.random.randint(low=self.image_folder_ranges[0], high=self.image_folder_ranges[1])

		archive = zipfile.ZipFile(self.image_path + '{:05d}.zip'.format(zip_index), 'r')

		imgs = []
		for iimg in range(2000):
			img_tag = '{:05d}-{:04}.jpg'.format(zip_index, iimg)

			img = image.load_img(archive.open(img_tag), target_size=(112, 112), color_mode='grayscale')
			img = np.squeeze(image.img_to_array(img))

			img -= np.mean(img)
			img += 128  # recenter image to be at 128, otherwise some images are too dark

			imgs.append(img)
		imgs = np.asarray(imgs)

		return imgs


	def get_white_noise_image(self):
		# get white noise image

		img = np.random.uniform(size=(1,112,112)) * 50 + 128


		return img


	def get_random_natural_images(self, num_random_images=1):
		# get random natural image in range 0 to 255

		imgs = []
		for iimg in range(num_random_images):
			if self.random_images.size == 0 or self.index_random_images == 2000:
				self.random_images = self.get_2k_random_images()
				self.index_random_images = 0
			
			imgs.append(self.random_images[self.index_random_images,:,:])

			self.index_random_images += 1

		return np.asarray(imgs)



	def get_gaudy_images(self, imgs):

		imgs_gaudy = np.copy(imgs)
		
		num_images = imgs.shape[0]
		for iimg in range(num_images):
			img = imgs_gaudy[iimg]

			m = np.mean(img)
			img[img < m] = 0
			img[img >= m] = 255

			imgs_gaudy[iimg] = img

		return imgs_gaudy


