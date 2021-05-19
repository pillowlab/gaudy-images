import numpy as np

import class_random_selection
import class_gaudi

from numpy.random import seed
from tensorflow import set_random_seed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



### frac var vs. session number for different AL algos (for main paper, no errorbars)
if False:

	f = plt.figure()

	pretrained_CNN = 'VGG19'

	algos = ['random_selection', 'gaudi', 'largebank_normal', 'synthetic_normal', 'coreset_normal']
	for algo in algos:

		corrs_test = np.load('./results/corrs_test_' + pretrained_CNN + '_' + algo + '.npy')

		plt.plot(corrs_test**2, label=algo)

	plt.xlabel('session number')
	plt.ylabel('frac var explained')
	plt.legend()

	f.savefig('./figs/frac_var_vs_session_number.pdf')



# TESTER CODE
### frac var vs. session number for different AL algos (for main paper, no errorbars)
if False:

	for pretrained_CNN in ['VGG19', 'InceptionV3', 'Densenet169']:
		f = plt.figure()

		pretrained_CNN = 'InceptionV3'
		irun = 0

		algos = ['random_selection', 'gaudi', 'largebank_normal', 'synthetic_normal', 'coreset_normal', 'largebank_gaudi', 'synthetic_gaudi', 'coreset_gaudy']
		for algo in algos:

			corrs_test = np.load('./results/corrs_test_' + pretrained_CNN + '_' + algo + '_run{:d}.npy'.format(irun))

			plt.plot(corrs_test**2, label=algo)

		plt.xlabel('session number')
		plt.ylabel('frac var explained')
		plt.legend()

		f.savefig('./figs/tester.pdf')



### frac var vs. session number for different AL algos (for supplemental, errorbars)
if False:

	algos = ['random_selection', 'largebank_normal', 'synthetic_normal', 'coreset_normal', 'largebank_gaudi', 'synthetic_gaudi', 'coreset_gaudy', 'gaudi']
		
	for pretrained_CNN in ['VGG19', 'InceptionV3', 'Densenet169']:

		f = plt.figure()

		num_runs = 5

		frac_vars = np.empty((num_runs, len(algos)))
		frac_vars[:] = np.nan

		for irun in range(num_runs):
			ialgo = 0
			for algo in algos:

				try:
					corrs_test = np.load('./results/corrs_test_' + pretrained_CNN + '_' + algo + '_run{:d}.npy'.format(irun))

					frac_vars[irun, ialgo] = corrs_test[-1]**2
				except:
					pass

				ialgo += 1

		print(frac_vars)

		plt.bar(np.arange(len(algos)), np.nanmean(frac_vars,axis=0))
		plt.errorbar(x=np.arange(len(algos)), y=np.nanmean(frac_vars,axis=0), yerr=np.nanstd(frac_vars,axis=0), fmt='none')
		plt.ylabel('frac var explained')

		plt.xticks(np.arange(len(algos)), labels=algos, rotation=60)

		f.savefig('./figs/errorbars_diff_AL_algos_{:s}.pdf'.format(pretrained_CNN), bbox_inches='tight')


### frac var vs num ensemble networks
if False:
	f = plt.figure()


	pretrained_CNN = 'VGG19'

	nums_ensemble_networks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
	num_runs = 5

	frac_vars = np.zeros((num_runs, len(nums_ensemble_networks)))
	for irun in range(num_runs):
		inum = 0
		for num_ensemble_networks in nums_ensemble_networks:

			corrs_test = np.load('./results/corrs_test_' + pretrained_CNN + '_random_selection_' + str(num_ensemble_networks) + 'networks_run{:d}.npy'.format(irun))

			frac_vars[irun, inum] = corrs_test[-1]**2
			inum += 1

	print(frac_vars)
	plt.errorbar(x=nums_ensemble_networks, y=np.mean(frac_vars, axis=0), yerr=np.std(frac_vars,axis=0))

	plt.xlabel('number ensemble networks')
	plt.ylabel('frac var explained')

	f.savefig('./figs/frac_var_vs_number_ensemble_networks.pdf')




### hyperparam search for synthetic gaudi
if True:

	f = plt.figure()

	pretrained_CNN = 'InceptionV3'

	algos = ['random_selection', 'gaudi']
	for algo in algos:

		corrs_test = np.load('./results/corrs_test_' + pretrained_CNN + '_' + algo + '_run0.npy')
		plt.plot(corrs_test**2, label=algo)

	for ihyperparam_set in np.arange(5):
		corrs_test = np.load('./results/corrs_synthetic_gaudi/corrs_test_' + pretrained_CNN + '_' + str(ihyperparam_set) + 'hyperparam_set.npy')
		plt.plot(corrs_test**2, label='{:d}hyperparam_set'.format(ihyperparam_set))

		print(corrs_test[-1]**2)
	plt.xlabel('session number')
	plt.ylabel('frac var explained')
	plt.legend()

	f.savefig('./figs/hyperparam_search_synthetic_gaudi.pdf')
