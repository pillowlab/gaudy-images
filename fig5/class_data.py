# modified script to handle num repeats per day


import numpy as np

import pickle

class DataClass:

	def __init__(self):
		self.data_folder_path = '/usr/people/bcowley/adroit/experiment_wilee/data/'


	def split_responses(self, responses):
		# responses: (num_neurons x num_stimuli x num_trials). (some elements are NAN b/c not all stimuli had same number of trials)

		num_neurons = responses.shape[0]
		num_images = responses.shape[1]

		responses1 = np.zeros(shape=(num_neurons,num_images))
		responses2 = np.zeros(shape=(num_neurons,num_images))

		mean_rate = np.mean(responses,axis=0)
		for iimage in range(num_images):
			num_repeats = np.sum(~np.isnan(mean_rate[iimage]))
			r = np.random.permutation(num_repeats)

			num_half = int(np.floor(num_repeats/2.))

			responses1[:,iimage] = np.mean(responses[:,iimage,r[:num_half]], axis=-1)
			responses2[:,iimage] = np.mean(responses[:,iimage,r[num_half:2*num_half]], axis=-1)

		return responses1, responses2


	def get_images_and_split_responses(self, day_id, I):
		# returns raw images and split responses
		# I: image class

		# get images
		if True:
			if day_id == 190923: # heldout day
				zip_filename = self.data_folder_path + 'images_heldout/images_190923.zip'
			else:  # a training day
				zip_filename = self.data_folder_path + 'images/zipped_jpg_files/images_{:d}.zip'.format(day_id)

			imgs = I.get_images_from_zipfile(zip_filename, np.arange(1,1201)) # get 1200 images


		# get responses
		if True:
			if day_id == 190923: # heldout day
				response_filename = self.data_folder_path + 'responses_heldout/raw/responses_190923.npy'
			else:  # a training day
				response_filename = self.data_folder_path + 'responses/raw/responses_{:d}.npy'.format(day_id)

			responses = np.load(response_filename)

			# discard images and responses if less than 10 repeats *or* image is synthetic
			num_images = responses.shape[1]
			num_max_repeats = responses.shape[2]
			inds_images_to_keep = []
			mean_rate = np.mean(responses,axis=0)
			for iimage in range(num_images):
				if np.sum(~np.isnan(mean_rate[iimage,:])) >= 10:
					if day_id != 190923 and (iimage < 600 or iimage >= 900): # second part removes synthetic images for training days
						inds_images_to_keep.append(iimage)
					elif day_id == 190923:
						inds_images_to_keep.append(iimage)

			imgs = imgs[inds_images_to_keep]
			responses = responses[:,inds_images_to_keep,:]

			# split responses
			responses1, responses2 = self.split_responses(responses)

		return imgs, responses1, responses2



	def get_images_and_averaged_responses(self, day_id, I):
		# returns raw images and repeat-averaged responses
		# I: image class

		# get images
		if True:
			if day_id == 190923: # heldout day
				zip_filename = self.data_folder_path + 'images_heldout/images_190923.zip'
			else:  # a training day
				zip_filename = self.data_folder_path + 'images/zipped_jpg_files/images_{:d}.zip'.format(day_id)

			imgs = I.get_images_from_zipfile(zip_filename, np.arange(1,1201)) # get 1200 images


		# get responses
		if True:
			if day_id == 190923: # heldout day
				response_filename = self.data_folder_path + 'responses_heldout/raw/responses_190923.npy'
			else:  # a training day
				response_filename = self.data_folder_path + 'responses/raw/responses_{:d}.npy'.format(day_id)

			responses = np.load(response_filename)

			# discard images and responses if less than 10 repeats *or* image is synthetic
			num_images = responses.shape[1]
			num_max_repeats = responses.shape[2]
			inds_images_to_keep = []
			mean_rate = np.mean(responses,axis=0)
			for iimage in range(num_images):
				if np.sum(~np.isnan(mean_rate[iimage,:])) >= 10:
					if day_id != 190923 and (iimage < 600 or iimage >= 900): # second part removes synthetic images for training days
						inds_images_to_keep.append(iimage)
					elif day_id == 190923:
						inds_images_to_keep.append(iimage)

			imgs = imgs[inds_images_to_keep]
			responses = responses[:,inds_images_to_keep,:]

			# average responses (over all repeats)
			responses = np.nanmean(responses, axis=-1)

		return imgs, responses






	# def get_images_responses_for_training(self, I, current_session_tag, AL_algo_name):
	# 	# returns images_train, responses_train, images_val, responses_val

	# 	filename = self.data_folder_path + 'images/metadata/metadata_' + current_session_tag + '.pkl'

	# 	with open(filename, 'rb') as fp:
	# 		metadata = pickle.load(fp)  # returns dict{'labels'}
	# 	labels = metadata['labels']
	# 	num_images = len(labels)

	# 	inds_images = []
	# 	inds_images_val = []
	# 	for iimage in range(num_images):
	# 		if AL_algo_name == 'random_selection':
	# 			if labels[iimage] == 'random_selection' or labels[iimage] == 'validation':
	# 				inds_images.append(iimage)
	# 		else:
	# 			if labels[iimage] == AL_algo_name or labels[iimage] == 'random_selection':
	# 				inds_images.append(iimage)

	# 		if labels[iimage] == 'validation':
	# 			inds_images_val.append(iimage)

	# 	zip_filename = self.data_folder_path + 'images/zipped_jpg_files/images_' + current_session_tag + '.zip'
	# 	imgs = I.get_images_from_zipfile(zip_filename, np.asarray(inds_images) + 1)
	# 	imgs_train = I.transform_images_to_resnet_format(imgs)

	# 	imgs = I.get_images_from_zipfile(zip_filename, np.asarray(inds_images_val) + 1)
	# 	imgs_val = I.transform_images_to_resnet_format(imgs)

	# 	# load mean responses
	# 	response_filename = self.data_folder_path + 'responses/averaged_over_repeats/responses_' + current_session_tag + '.npy'

	# 	responses = np.load(response_filename) # returns matrix (num_neurons x num_images)

	# 	responses_train = responses[:,inds_images]
	# 	responses_val = responses[:,inds_images_val]


	# 	# if any responses are NaN => need to remove them and corresponding images
	# 	indices = np.isnan(np.sum(responses_train, axis=0)) == False

	# 	imgs_train = imgs_train[indices]
	# 	responses_train = responses_train[:,indices]

	# 	indices = np.isnan(np.sum(responses_val, axis=0)) == False

	# 	imgs_val = imgs_val[indices]
	# 	responses_val = responses_val[:,indices]

	# 	return imgs_train, responses_train, imgs_val, responses_val


	# def get_images_responses(self, I, current_session_tag, AL_algo_name):
	# 	# returns images, responses for particular algorithm (300 images each)
	# 	#	algo_names = {'random_selection', 'largebank_ensvar', 'natprior_ensvar'}

	# 	filename = self.data_folder_path + 'images/metadata/metadata_' + current_session_tag + '.pkl'

	# 	with open(filename, 'rb') as fp:
	# 		metadata = pickle.load(fp)  # returns dict{'labels'}
	# 	labels = metadata['labels']
	# 	num_images = len(labels)

	# 	inds_images = []
	# 	inds_images_val = []
	# 	for iimage in range(num_images):
	# 		if labels[iimage] == AL_algo_name:
	# 			inds_images.append(iimage)

	# 	zip_filename = self.data_folder_path + 'images/zipped_jpg_files/images_' + current_session_tag + '.zip'
	# 	imgs = I.get_images_from_zipfile(zip_filename, np.asarray(inds_images) + 1)
	# 	imgs = I.transform_images_to_resnet_format(imgs)

	# 	# load mean responses
	# 	response_filename = self.data_folder_path + 'responses/averaged_over_repeats/responses_' + current_session_tag + '.npy'

	# 	responses = np.load(response_filename) # returns matrix (num_neurons x num_images)
	# 	responses = responses[:,inds_images]

	# 	# if any responses are NaN => need to remove them and corresponding images
	# 	indices = np.isnan(np.sum(responses, axis=0)) == False

	# 	imgs = imgs[indices]
	# 	responses = responses[:,indices]

	# 	return imgs, responses


	# def get_all_images_and_responses_from_session(self, I, session_tag, num_repeats=0):
	# 	# imgs will be in ResNet format
		
	# 	filename = self.data_folder_path + 'images/metadata/metadata_' + session_tag + '.pkl'

	# 	with open(filename, 'rb') as fp:
	# 		metadata = pickle.load(fp)  # returns dict{'labels'}
	# 	labels = metadata['labels']
	# 	num_images = len(labels)

	# 	inds_images = np.arange(num_images)

	# 	zip_filename = self.data_folder_path + 'images/zipped_jpg_files/images_' + session_tag + '.zip'
	# 	imgs = I.get_images_from_zipfile(zip_filename, inds_images + 1)
	# 	imgs = I.transform_images_to_resnet_format(imgs)

	# 	# load raw responses
	# 	response_filename = self.data_folder_path + 'responses/raw/responses_' + session_tag + '.npy'

	# 	responses_raw = np.load(response_filename)
	# 		# (num_neurons, num_images, num_repeats)

	# 	num_neurons = responses_raw.shape[0]

	# 	# take average responses, only up to num_repeats
	# 	responses = np.zeros((num_neurons, num_images))
	# 	np.random.seed(31490)  # makes sure same order is returned every time
	# 	for iimage in range(num_images):
	# 		num_total_repeats = np.sum(~np.isnan(responses_raw[0,iimage]))
			
	# 		if num_repeats == 0:  # use all repeats
	# 			responses[:,iimage] = np.mean(responses_raw[:,iimage,:num_total_repeats],axis=-1)
	# 		else:
	# 			r = np.random.permutation(num_total_repeats)
	# 			responses[:,iimage] = np.mean(responses_raw[:,iimage,r[:num_repeats]],axis=-1)

	# 	# if any responses are NaN => need to remove them and corresponding images
	# 	indices = np.isnan(np.sum(responses, axis=0)) == False

	# 	imgs = imgs[indices]
	# 	responses = responses[:,indices]

	# 	return imgs, responses


	# def get_heldout_images_and_responses(self, I, transformToResnet=True):

	# 	# get images
	# 	zip_filename = self.data_folder_path + 'images_heldout/images_190923.zip'
	# 	# zip_filename = '../data/images/zipped_jpg_files/images_190924.zip'
	# 	imgs = I.get_images_from_zipfile(zip_filename, np.arange(1,1201)) # get 1200 images
	# 	if transformToResnet == True:
	# 		imgs = I.transform_images_to_resnet_format(imgs)

	# 	# get responses
	# 	response_filename = self.data_folder_path + 'responses_heldout/raw/responses_190923.npy'
	# 	# response_filename = '../data/responses/raw/responses_190924.npy'

	# 	responses = np.load(response_filename) # returns matrix (num_neurons x num_images)

	# 	# remove images with less than 10 repeats
	# 	num_images = responses.shape[1]
	# 	num_max_repeats = responses.shape[2]
	# 	inds_images_to_keep = []
	# 	mean_rate = np.mean(responses,axis=0)
	# 	for iimage in range(num_images):
	# 		if np.sum(~np.isnan(mean_rate[iimage,:])) >= 10:
	# 			inds_images_to_keep.append(iimage)


	# 	imgs = imgs[inds_images_to_keep]
	# 	responses = responses[:,inds_images_to_keep,:]

	# 	return imgs, responses









