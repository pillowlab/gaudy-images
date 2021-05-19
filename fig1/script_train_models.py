

import numpy as np

import class_images
import class_gabor_responses

import class_linear_model
import class_relu_model
import class_sigmoid_model

import class_random_selection
import class_gaudi_images

from numpy.random import seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys



def get_test_set(I, R):

	imgs = I.get_random_natural_images(num_random_images=4000)

	imgs_test = imgs
	responses_test = R.get_responses_from_imgs(imgs)

	return imgs_test, responses_test


def run_model(I, R, M, A, save_filetag, num_sessions, num_epochs=5, seed_index=50):

	frac_vars_test = []

	for isession in range(num_sessions):

		imgs_train = A.get_images_to_show(I)
		responses_train = R.get_responses_from_imgs(imgs_train)

		print('session: ' + str(isession))

		# if isession == 0:
		# 	f = plt.figure(figsize=(15,15))
		# 	for iimg in range(25):
		# 		plt.subplot(5,5,iimg+1)
		# 		plt.imshow(imgs_train[iimg]/255, cmap='gray')
		# 		plt.axis('off')
		# 	f.savefig('./figs/tester2.pdf')
		# 	plt.close('all')

		for iepoch in range(num_epochs):
			M.train_model(imgs_train, responses_train)

			frac_var_test = M.compute_frac_var(imgs_test, responses_test)

			print('         epoch {:d}: frac_var_test = {:0.4f}'.format(iepoch, frac_var_test))

		frac_vars_test.append(frac_var_test)

		np.save('./results/frac_vars_' + save_filetag + '.npy', np.asarray(frac_vars_test))



### MAIN SCRIPT

num_sessions = 100
num_epochs = 5
seed_index = 50


# inputs:
#  python -u script_train_models.py gpu_id model algo run_id

model = sys.argv[2]  # {'linear', 'relu', 'sigmoid'}
algo = sys.argv[3] # {'random_selection', 'gaudi_images'}
run_id = int(sys.argv[4])

seed_index = 100 * run_id  # change seed index for each run

# get models set up
seed(seed_index)

if model == 'linear':
	M = class_linear_model.LinearModel()   
	R = class_gabor_responses.GaborResponsesClass(passthrough_function='linear')
elif model == 'relu':
	M = class_relu_model.ReluModel() 
	R = class_gabor_responses.GaborResponsesClass(passthrough_function='relu')
elif model == 'sigmoid':
	M = class_sigmoid_model.SigmoidModel() 
	R = class_gabor_responses.GaborResponsesClass(passthrough_function='sigmoid')

I = class_images.ImageClass()


# get test set
seed(50)  # ensure test set is same across runs
imgs_test, responses_test = get_test_set(I, R)
seed(seed_index)

if algo == 'random_selection':
	M.initialize_weights(seed_index)
	A = class_random_selection.RandomSelectionClass()
	run_model(I, R, M, A, 'random_selection_{:s}_run{:d}'.format(model, run_id), num_sessions, num_epochs, seed_index)

	# f = plt.figure()
	# k = M.get_kernel()
	# plt.imshow(k, cmap='gray')
	# f.savefig('./figs/kernel_{:s}_randsel_run{:d}.pdf'.format(model, run_id))


if algo == 'gaudi_images':
	M.initialize_weights(seed_index)
	A = class_gaudi_images.GaudiImageClass()
	run_model(I, R, M, A, 'gaudi_images_{:s}_run{:d}'.format(model, run_id), num_sessions, num_epochs, seed_index)

	# f = plt.figure()
	# k = M.get_kernel()
	# plt.imshow(k, cmap='gray')
	# f.savefig('./figs/kernel_{:s}_gaudi_run{:d}.pdf'.format(model, run_id))





