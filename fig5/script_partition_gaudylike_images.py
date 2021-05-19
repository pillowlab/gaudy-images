

import numpy as np

import class_data
import class_images

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




### HELPER FUNCTIONS

def get_gaudy_metrics(imgs_raw):
	# returns vector of values of gaudy metric: high --> image is far from its gaudy version, low --> image is close to its gaudy version

	mses_gaudy_dissimilarity = []

	num_imgs = imgs_raw.shape[0]
	imgs_gaudy = np.copy(imgs_raw)
	for iimg in range(num_imgs):
		img = imgs_gaudy[iimg]
		m = np.mean(img)
		img[img < m] = 0
		img[img >= m] = 255
		imgs_gaudy[iimg] = img

		# take mse
		mses_gaudy_dissimilarity.append(np.mean((imgs_raw[iimg] - imgs_gaudy[iimg])**2))

	return mses_gaudy_dissimilarity



def get_inds_normal(num_normal_imgs, num_total_imgs):

	r = np.random.permutation(num_total_imgs)
	return r[:num_normal_imgs]


### MAIN SCRIPT

D = class_data.DataClass()
I = class_images.ImageClass()

day_ids = [190924, 190925, 190926, 190927, 190928, 190929]


## sort images based on gaudiness 
#		gaudylike images: 200 most gaudylike, then 200 randomly chosen from remaining images
#		normal images:  400 randomly-chosen images
if True:
	for day_id in day_ids:

		print(day_id)

		imgs, responses = D.get_images_and_averaged_responses(day_id, I)

		metrics = get_gaudy_metrics(imgs)
		
		# get gaudylike images
		inds_sort = np.argsort(metrics)
		num_total_imgs = inds_sort.size

		inds_gaudylike = inds_sort[:400]
		inds_nongaudylike = inds_sort[-400:]

		inds_400gaudy = inds_gaudylike
		inds_300gaudy_100normal = np.concatenate([inds_gaudylike[:300], get_inds_normal(100,num_total_imgs)])
		inds_200gaudy_200normal = np.concatenate([inds_gaudylike[:200], get_inds_normal(200,num_total_imgs)])
		inds_400normal = get_inds_normal(400,num_total_imgs)
		inds_200nongaudy_200normal = np.concatenate([inds_nongaudylike[:200], get_inds_normal(200,num_total_imgs)])
		inds_300nongaudy_100normal = np.concatenate([inds_nongaudylike[:300], get_inds_normal(100,num_total_imgs)])
		inds_400nongaudy = inds_nongaudylike


		np.save('./results/inds/inds_400gaudy.npy', inds_400gaudy)
		np.save('./results/inds/inds_300gaudy_100normal.npy', inds_300gaudy_100normal)
		np.save('./results/inds/inds_200gaudy_200normal.npy', inds_200gaudy_200normal)
		np.save('./results/inds/inds_400normal.npy', inds_400normal)
		np.save('./results/inds/inds_200nongaudy_200normal.npy', inds_200nongaudy_200normal)
		np.save('./results/inds/inds_300nongaudy_100normal.npy', inds_300nongaudy_100normal)
		np.save('./results/inds/inds_400nongaudy.npy', inds_400nongaudy)


## plot images that are gaudylike/nongaudylike for each day
if False:
	for day_id in day_ids:

		# plot gaudylike images
		imgs_gaudylike = np.load('./results/images/imgs_gaudylike_day{:d}.npy'.format(day_id))

		f = plt.figure()
		for iimg in range(16):
			plt.subplot(4,4,iimg+1)
			plt.imshow(imgs_gaudylike[iimg].astype('uint8'))
			plt.axis('off')

		f.tight_layout()
		f.savefig('./figs/images_gaudylike_day_{:d}.pdf'.format(day_id))


		# plot normal images
		imgs_normal = np.load('./results/images/imgs_normal_day{:d}.npy'.format(day_id))

		f = plt.figure()
		for iimg in range(16):
			plt.subplot(4,4,iimg+1)
			plt.imshow(imgs_normal[iimg].astype('uint8'))
			plt.axis('off')

		f.tight_layout()
		f.savefig('./figs/images_normal_day_{:d}.pdf'.format(day_id))






