
import numpy as np

import class_model
import class_images
import class_features
import class_surrogate_responses

import class_random_selection
import class_gaudi


from numpy.random import seed
from tensorflow import set_random_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import sys


### HELPER FUNCTIONS


def get_test_set(I, F, R):

	imgs = I.get_random_natural_images(num_random_images=4000)

	features_test = F.get_features_from_imgs(imgs)
	responses_test = R.get_responses_from_imgs(imgs)

	return features_test, responses_test


def get_initial_train_set(I, F, R):

	imgs = I.get_random_natural_images(num_random_images=2000)

	features_test = F.get_features_from_imgs(imgs)
	responses_test = R.get_responses_from_imgs(imgs)

	return features_test, responses_test

def plot_images(imgs, errornorms, save_filetag, isession):
	num_images_to_show = 25
	num_total_images = imgs.shape[0]
	ind_sorted = np.argsort(errornorms)

	ind_sorted = ind_sorted[np.arange(0,num_total_images,np.ceil(num_total_images/num_images_to_show)).astype(int)]

	f = plt.figure(figsize=(10,10))

	panel_index = 1

	for ind in ind_sorted:

		plt.subplot(5,5,panel_index)

		img = imgs[ind]
		plt.imshow(img/255)
		plt.axis('off')
		plt.title('{:0.3f}'.format(errornorms[ind]))
		panel_index += 1

	f.savefig('./figs/' + save_filetag + '/images_session' + str(isession) + '.pdf')


def plot_pcs(responses_test, responses_AL, save_filetag, isession):

	f = plt.figure(figsize=(15,15))

	plt.subplot(2,2,1)
	pca = PCA(n_components=2)
	pca.fit(np.hstack((responses_test, responses_AL)).T)

	Z_test = pca.transform(responses_test.T).T
	Z_AL = pca.transform(responses_AL.T).T

	plt.plot(Z_test[0,:], Z_test[1,:], '.k', alpha=0.2, label='test')
	plt.plot(Z_AL[0,:], Z_AL[1,:], '.r', alpha=0.2, label='AL')

	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.legend()
	plt.title('response space')

	plt.subplot(2,2,2)
	pca = PCA(n_components=15)

	pca.fit(responses_test.T)
	plt.plot(pca.explained_variance_, 'k', alpha=0.2, label='test')

	pca.fit(responses_AL.T)
	plt.plot(pca.explained_variance_, 'r', alpha=0.2, label='AL')

	plt.xlabel('PC index')
	plt.ylabel('variance explained')
	plt.legend()
	plt.title('response space')

	f.savefig('./figs/' + save_filetag + '/pcs_session' + str(isession) + '.pdf')


def plot_errornorm_histograms(errornorms_test, errornorms_AL, save_filetag, isession):

	f = plt.figure()

	plt.hist(errornorms_test, color='k', alpha=0.5, label='test')

	plt.hist(errornorms_AL, color='r', alpha=0.5, label='AL')

	plt.xlabel('errornorm')
	plt.ylabel('count')
	plt.legend()

	f.savefig('./figs/' + save_filetag + '/errornorms_session' + str(isession) + '.pdf')


def run_model(I, F, R, M, A, pretrained_CNN, save_filetag, run_id, num_sessions, seed_index=50):

	frac_vars_test = []

	print('preparing to run ' + save_filetag + '...')

	for isession in range(num_sessions):

		imgs_train = A.get_images_to_show(I, F, R, M)

		features_train = F.get_features_from_imgs(imgs_train)
		responses_train = R.get_responses_from_imgs(imgs_train)

		if False:
			save_path = pretrained_CNN + '/' + save_filetag

			responses_train_hat = M.get_predicted_ensemble_responses(features_train)
			errornorms = np.sqrt(np.sum((responses_train - responses_train_hat)**2,axis=0))

			plot_images(imgs_train, errornorms, save_path, isession)

			responses_test_hat = M.get_predicted_ensemble_responses(features_test[:500,:,:,:])
			plot_pcs(responses_test_hat, responses_train_hat, save_path, isession)

			errornorms_test = np.sqrt(np.sum((responses_test[:,:500] - responses_test_hat)**2,axis=0))
			plot_errornorm_histograms(errornorms_test, errornorms, save_path, isession)
			
			plt.close('all')


		print('session: ' + str(isession))
		print('   num chosen images: {:d}'.format(imgs_train.shape[0]))

		for iepoch in range(num_epochs):

			M.train_models(features_train, responses_train)

			frac_var_train = M.compute_frac_var(features_train, responses_train)

			frac_var_val = M.compute_frac_var(features_test, responses_test)

			print('         epoch {:d}: frac_var_val = {:0.4f}, frac_var_train = {:0.4f}'.format(iepoch, 
																frac_var_val, frac_var_train))

		frac_var_test = frac_var_val  # same computation
		frac_vars_test.append(frac_var_test)

		print('{:s}: frac_var_test = {:0.4f}'.format(save_filetag, frac_var_test))

		np.save('./results/frac_vars_test_' + pretrained_CNN + '_' + save_filetag + '_run{:d}.npy'.format(run_id), np.asarray(frac_vars_test))




### MAIN SCRIPT

# to run:
#	python -u script_train_model.py #gpu_id $pretrained_CNN $image_type $run_id > output_$pretrainedCNN_$image_type.txt

pretrained_CNN = sys.argv[2]  # {'VGG19', 'InceptionV3', 'Densenet169'}
images_type = sys.argv[3]
run_id = int(sys.argv[4])

num_sessions = 30
num_epochs = 5
num_models = 1

seed_index = 100 * run_id;

# get models set up
seed(seed_index)
set_random_seed(seed_index)

if pretrained_CNN == 'VGG19':
	learning_rate = 5e-1
elif pretrained_CNN == 'InceptionV3':
	learning_rate = 1e0 #1e-1
elif pretrained_CNN == 'Densenet169':
	learning_rate = 1e0 #2e-1
	
M = class_model.ModelEnsembleClass(num_models=num_models, num_output_vars=100, learning_rate=learning_rate, seed_index=seed_index)   # goes first for GPU allocation

I = class_images.ImageClass()
F = class_features.ResnetFeaturesClass()
R = class_surrogate_responses.SurrogateResponsesClass(pretrained_CNN=pretrained_CNN)

print('models loaded.')

print('getting test set...')
features_test, responses_test = get_test_set(I,F,R)

# # train initial model with 2k images
# if images_type == 'random_selection':
# 	# features_train, responses_train = get_initial_train_set(I,F,R)
# 	# for iepoch in range(5):
# 	# 	M.train_models(features_train, responses_train)
# 	M.save_models('initial_training_{:s}_run{:d}'.format(pretrained_CNN, run_id))
# 	print('initial model trained')
# else:
# 	M.load_models('initial_training_{:s}_run{:d}'.format(pretrained_CNN, run_id))


### random selection
if images_type == 'random_selection':
	A = class_random_selection.RandomSelection()
	run_model(I, F, R, M, A, pretrained_CNN, 'random_selection', run_id, num_sessions)

### Gaudi images
if images_type == 'gaudi':
	A = class_gaudi.GaudiSelectionClass()
	run_model(I, F, R, M, A, pretrained_CNN, 'gaudi', run_id, num_sessions)


