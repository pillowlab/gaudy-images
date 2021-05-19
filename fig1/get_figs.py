

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### plot results for run0 (to test models)
if False:
	models = ['linear']
	algos = ['random_selection', 'gaudi_images']

	for irun in [0,1,2,3,4]:

		for model in models:

			f = plt.figure()

			for algo in algos:

				frac_vars = np.load('./results/frac_vars_' + algo + '_' + model + '_run' + str(irun) + '.npy')

				plt.plot(np.squeeze(frac_vars), label=algo)

			plt.ylabel('frac var explained')
			plt.xlabel('session number')

			plt.legend()

			f.savefig('./figs/frac_vars_{:s}_model_run{:d}.pdf'.format(model, irun))


### plot mean and std over runs (30 sessions)
if True:
	models = ['linear', 'relu', 'sigmoid']
	algos = ['random_selection', 'gaudi_images']
	nums_runs = {'linear':5, 'relu':5, 'sigmoid':5}
	num_sessions = 30

	for model in models:

		f = plt.figure()

		for algo in algos:
			fvs = [];
			for irun in range(nums_runs[model]):
				frac_vars = np.load('./results/frac_vars_' + algo + '_' + model + '_run' + str(irun) + '.npy')

				fvs.append(frac_vars)


			fvs = np.array(fvs)

			m = np.mean(fvs, axis=0)
			s = np.std(fvs, axis=0)
			plt.errorbar(x=np.arange(num_sessions), y=m[:num_sessions], yerr=s[:num_sessions], label=algo)

		plt.ylabel('frac var explained')
		plt.xlabel('session number')

		plt.legend()

		f.savefig('./figs/frac_vars_{:s}_model.pdf'.format(model))



### plot mean and std over runs (100 sessions)
if True:
	models = ['linear', 'relu', 'sigmoid']
	algos = ['random_selection', 'gaudi_images']
	nums_runs = {'linear':5, 'relu':5, 'sigmoid':5}

	for model in models:

		f = plt.figure()

		for algo in algos:
			fvs = [];
			for irun in range(nums_runs[model]):
				frac_vars = np.load('./results/frac_vars_' + algo + '_' + model + '_run' + str(irun) + '.npy')

				fvs.append(frac_vars)


			fvs = np.array(fvs)

			m = np.mean(fvs, axis=0)
			s = np.std(fvs, axis=0)
			plt.errorbar(x=np.arange(m.size), y=m, yerr=s, label=algo)

		plt.ylabel('frac var explained')
		plt.xlabel('session number')

		plt.legend()

		f.savefig('./figs/frac_vars_{:s}_model_100sessions.pdf'.format(model))

