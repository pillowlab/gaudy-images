import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### plot for 30 sessions
if False:
	pretrained_CNNs = ['VGG19', 'InceptionV3', 'Densenet169']
	num_runs = 5
	num_sessions = 30

	f = plt.figure(figsize=(10,10))
	ipanel = 1

	for pretrained_CNN in pretrained_CNNs:

		plt.subplot(2,2,ipanel)

		for image_type in ['random_selection', 'gaudi']:
			frac_vars_runs = np.zeros((num_runs, num_sessions))

			for irun in range(num_runs):
				frac_vars_runs[irun] = np.load('./results/corrs_test_' + pretrained_CNN + '_' + image_type + '_run{:d}.npy'.format(irun))**2

			x = np.arange(1,num_sessions+1)
			y = np.mean(frac_vars_runs,axis=0)
			yerr = np.std(frac_vars_runs, axis=0)
			plt.errorbar(x=x, y=y, yerr=yerr, label=image_type)
			
		plt.xlabel('session number')
		plt.ylabel('frac var explained')
		plt.title(pretrained_CNN)
		
		ipanel += 1


	f.savefig('./figs/frac_vars_vs_session_number.pdf', transparent=True)


### plot for all sessions
if True:
	pretrained_CNNs = ['VGG19', 'InceptionV3', 'Densenet169']
	num_runs = 5
	num_sessions = 100

	f = plt.figure(figsize=(10,10))
	ipanel = 1

	for pretrained_CNN in pretrained_CNNs:

		plt.subplot(2,2,ipanel)

		for image_type in ['random_selection', 'gaudi']:
			frac_vars_runs = np.zeros((num_runs, num_sessions))

			for irun in range(num_runs):
				frac_vars_runs[irun] = np.load('./results/frac_vars_test_' + pretrained_CNN + '_' + image_type + '_run{:d}.npy'.format(irun))

			x = np.arange(1,num_sessions+1)
			y = np.mean(frac_vars_runs,axis=0)
			yerr = np.std(frac_vars_runs, axis=0)
			plt.errorbar(x=x, y=y, yerr=yerr, label=image_type)
			
		plt.xlabel('session number')
		plt.ylabel('frac var explained')
		plt.title(pretrained_CNN)
		
		ipanel += 1


	f.savefig('./figs/frac_vars_vs_session_number_all_sessions.pdf', transparent=True)


