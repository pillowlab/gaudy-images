# idea:
#	first plot PCA in pixel space between normal + gaudy
#	then compute the objective function from AL theory
#	for both normal and gaudy image

import numpy as np

import class_images

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy import linalg

import sys


### HELPER FUNCTIONS


### MAIN SCRIPT

I = class_images.ImageClass()

num_images = 10000

imgs_normal = I.get_random_natural_images(num_random_images=num_images)
imgs_gaudy = I.get_gaudy_images(imgs_normal)

imgs_for_pca = I.get_random_natural_images(num_random_images=num_images)


X_normal = np.reshape(imgs_normal, (num_images,-1))
X_gaudy = np.reshape(imgs_gaudy, (num_images,-1))
X_for_pca = np.reshape(imgs_for_pca, (num_images,-1))


### PCA analysis
if True:
	pca = PCA(n_components=2000)

	pca.fit(X_for_pca)  # apply PCA to separate, heldout normal images

	Z = pca.transform(X_normal)
	v_normal = np.var(Z, axis=0)

	Z = pca.transform(X_gaudy)
	v_gaudy = np.var(Z, axis=0)

	f = plt.figure(figsize=(10,10))

	plt.subplot(2,2,1)

	plt.plot(np.log(v_normal), 'k')
	plt.plot(np.log(v_gaudy), 'r')

	plt.ylabel('variance (log)')
	plt.xlabel('PC index')

	plt.subplot(2,2,2)

	plt.plot(v_gaudy/v_normal, '.')
	plt.ylabel('var gaudy / var normal')
	plt.xlabel('PC index (PCA on pixel intensities)')

	f.savefig('./figs/pca_pixel_intensities.pdf')


### compute obj function for pixel intensities
if False:
	
	pca = PCA(n_components=2000)
	pca.fit(X_normal)

	U = pca.components_

	Z_normal = np.dot(U, X_normal.T)

	Sigma = np.cov(Z_normal)
	Sigma_inv = linalg.inv(Sigma)

	Sigma_inv_squared = Sigma_inv * Sigma_inv

	num_images = 500
	imgs_normal = I.get_random_natural_images(num_random_images=num_images)
	imgs_gaudy = I.get_gaudy_images(imgs_normal)

	X_normal = np.reshape(imgs_normal, (num_images,-1))
	X_gaudy = np.reshape(imgs_gaudy, (num_images,-1))

	Z_normal = np.dot(U, X_normal.T - pca.mean_[:,np.newaxis])
	Z_gaudy = np.dot(U, X_gaudy.T - pca.mean_[:,np.newaxis])
			# num vars x num samples

	obj_val_numerator = np.diagonal(np.dot(Z_normal.T, np.dot(Sigma_inv_squared, Z_normal)))
	obj_val_denominator = np.diagonal(1. + np.dot(Z_normal.T, np.dot(Sigma_inv, Z_normal)))

	obj_val_normal = obj_val_numerator / obj_val_denominator

	obj_val_numerator = np.diagonal(np.dot(Z_gaudy.T, np.dot(Sigma_inv_squared, Z_gaudy)))
	obj_val_denominator = np.diagonal(1. + np.dot(Z_gaudy.T, np.dot(Sigma_inv, Z_gaudy)))

	obj_val_gaudy = obj_val_numerator / obj_val_denominator


	f = plt.figure()

	minner = np.min(np.concatenate((obj_val_normal, obj_val_gaudy)))
	maxer = np.max(np.concatenate((obj_val_normal, obj_val_gaudy)))

	plt.plot(np.array([minner, maxer]), np.array([minner, maxer]), '--k')
	plt.plot(obj_val_normal, obj_val_gaudy, '.')

	plt.xlabel('obj val, normal images')
	plt.ylabel('obj val, gaudy images')

	f.savefig('./figs/obj_func_vals_normal_vs_gaudy.pdf')






