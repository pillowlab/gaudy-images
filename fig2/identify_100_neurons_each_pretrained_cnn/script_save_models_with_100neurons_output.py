

import numpy as np

import class_images
import class_features

from tensorflow.keras.layers import Lambda, Dense

from tensorflow.keras.models import Model

from tensorflow.keras import backend as K


### MAIN SCRIPT

for pretrained_CNN in ['ResNet50']: # ['ResNet50', VGG19', 'InceptionV3', 'Densenet169']:

	print('pretrained CNN: {:s}'.format(pretrained_CNN))

	# inds = np.load('./results/inds_highvar_{:s}.npy'.format(pretrained_CNN))

	# vars_sorted = np.load('./results/vars_sorted_highvar_{:s}.npy'.format(pretrained_CNN))
	# sigs_sorted = np.sqrt(vars_sorted)

	# means_sorted = np.load('./results/means_sorted_highvar_{:s}.npy'.format(pretrained_CNN))

	weights_matrix = np.load('./results/pca_weights_{:s}.npy'.format(pretrained_CNN))
	means = np.load('./results/pca_means_{:s}.npy'.format(pretrained_CNN))

	F = class_features.FeaturesClass()

	F.load_model(pretrained_CNN=pretrained_CNN)

	x = Dense(units=100, name='mask_layer')(F.model.output)  # use mask to get 100 top neurons
	F.model = Model(inputs=F.model.input, outputs=x)

	weights = F.model.get_layer('mask_layer').get_weights()
		# weights[0] contains the weight matrix

	weights[0] = weights_matrix.T
	weights[1] = -np.dot(weights_matrix,means)

	# weights[0] = np.zeros((weights[0].shape))
	# for iind in range(inds.size):
	# 	ind = inds[iind]

	# 	weights[0][ind,iind] = 1./sigs_sorted[iind]
	# 	weights[1][iind] = -means_sorted[iind] / sigs_sorted[iind]

	F.model.get_layer('mask_layer').set_weights(weights)

	F.model.save('./results/saved_models/model_{:s}.h5'.format(pretrained_CNN))


