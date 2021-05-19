# idea:
#  compute variance of all CNN neurons for layer
#	then order them and see the drop off...do all neurons have similar variance?


import numpy as np

import class_images
import class_features

from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### MAIN SCRIPT

I = class_images.ImageClass()

F = class_features.FeaturesClass()


np.random.seed(31490)
imgs_orig = I.get_random_natural_images(num_random_images=5000)


for pretrained_CNN in ['ResNet50']: #['ResNet50', 'VGG19', 'InceptionV3', 'Densenet169']:

	print('working on {:s}'.format(pretrained_CNN))

	F.load_model(pretrained_CNN=pretrained_CNN)

	imgs = np.copy(imgs_orig)
	features = F.get_features_from_imgs(imgs)

	pca = PCA(n_components=100)

	pca.fit(features)



	f = plt.figure()
	plt.plot(pca.explained_variance_, '.')

	plt.xlabel('neuron index (sorted)')
	plt.ylabel('variance over images')

	plt.title(pretrained_CNN)

	f.savefig('./figs/vars_sorted_{:s}.pdf'.format(pretrained_CNN))

	np.save('./results/pca_weights_{:s}.npy'.format(pretrained_CNN), pca.components_ / np.sqrt(pca.explained_variance_[:,np.newaxis]))
	np.save('./results/pca_means_{:s}.npy'.format(pretrained_CNN), pca.mean_)


	# # save CNN neurons with largest variance
	# inds_sorted = np.argsort(vs)

	# # # take top 100 neurons
	# # inds_sorted = inds_sorted[-100:]  # take top 100 neurons
	# # vars_sorted = vs_sorted[-100:]

	# # take random neurons (but var greater than 0)
	# inds_top_half = np.linspace(inds_sorted.size/2, inds_sorted.size-1, 100).astype(int)
	# inds_sorted = inds_sorted[inds_top_half]
	# vars_sorted = vs_sorted[inds_top_half]


	# means_sorted = np.mean(features[:,inds_sorted],axis=0)
	# np.save('./results/means_sorted_highvar_{:s}.npy'.format(pretrained_CNN), means_sorted)
	# 	# (num_neurons, num_images)

	# np.save('./results/vars_sorted_highvar_{:s}.npy'.format(pretrained_CNN), vars_sorted)
	# 	# (num_neurons, num_images)

	# np.save('./results/inds_highvar_{:s}.npy'.format(pretrained_CNN), inds_sorted)
	# 	# (100,), inds of CNN neurons to use

	# # save CNN neurons (randomly chosen)
	# r = np.random.permutation(features.shape[1])
	# responses = features[:,r[:100]].T
	# np.save('./results/responses_random_{:s}.npy'.format(pretrained_CNN), responses)
	# 	# (num_neurons, num_images)







# ### MAIN SCRIPT

# I = class_images.ImageClass()

# F = class_features.FeaturesClass()


# np.random.seed(31490)
# imgs_orig = I.get_random_natural_images(num_random_images=5000)


# for pretrained_CNN in ['VGG19', 'InceptionV3', 'Densenet169']:

# 	print('working on {:s}'.format(pretrained_CNN))

# 	F.load_model(pretrained_CNN=pretrained_CNN)

# 	imgs = np.copy(imgs_orig)
# 	features = F.get_features_from_imgs(imgs)

# 	vs = np.var(features,axis=0)

# 	vs_sorted = np.sort(vs)

# 	f = plt.figure()
# 	plt.plot(vs_sorted, '.')

# 	plt.xlabel('neuron index (sorted)')
# 	plt.ylabel('variance over images')

# 	plt.title(pretrained_CNN)

# 	f.savefig('./figs/vars_sorted_{:s}.pdf'.format(pretrained_CNN))

# 	# save CNN neurons with largest variance
# 	inds_sorted = np.argsort(vs)

# 	# # take top 100 neurons
# 	# inds_sorted = inds_sorted[-100:]  # take top 100 neurons
# 	# vars_sorted = vs_sorted[-100:]

# 	# take random neurons (but var greater than 0)
# 	inds_top_half = np.linspace(inds_sorted.size/2, inds_sorted.size-1, 100).astype(int)
# 	inds_sorted = inds_sorted[inds_top_half]
# 	vars_sorted = vs_sorted[inds_top_half]


# 	means_sorted = np.mean(features[:,inds_sorted],axis=0)
# 	np.save('./results/means_sorted_highvar_{:s}.npy'.format(pretrained_CNN), means_sorted)
# 		# (num_neurons, num_images)

# 	np.save('./results/vars_sorted_highvar_{:s}.npy'.format(pretrained_CNN), vars_sorted)
# 		# (num_neurons, num_images)

# 	np.save('./results/inds_highvar_{:s}.npy'.format(pretrained_CNN), inds_sorted)
# 		# (100,), inds of CNN neurons to use

# 	# # save CNN neurons (randomly chosen)
# 	# r = np.random.permutation(features.shape[1])
# 	# responses = features[:,r[:100]].T
# 	# np.save('./results/responses_random_{:s}.npy'.format(pretrained_CNN), responses)
# 	# 	# (num_neurons, num_images)











