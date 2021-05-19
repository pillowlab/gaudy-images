# handles vgg19, inceptionv3, and densenet169 model

import os
from os import walk
import numpy as np

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet169

from PIL import Image



class SurrogateResponsesClass:

 
### INTERNAL HELPER FUNCTIONS


### INIT FUNCTION

	def __init__(self, pretrained_CNN='VGG19'):
		# pretrained_CNNs: {'VGG19', 'InceptionV3', 'DenseNet169'}

		self.pretrained_CNN = pretrained_CNN

		# load VGG model (already maps images to surrogate neurons)
		saved_model_filename = '../fig2_dnns/identify_100_neurons_each_pretrained_cnn/results/saved_models/model_{:s}.h5'.format(pretrained_CNN)
		self.model = load_model(saved_model_filename)



### CALL FUNCTIONS

	def get_responses_from_imgs(self, imgs_raw):
		# assume imgs are raw, need to preprocess 

		imgs = np.copy(imgs_raw) # do not change imgs_raw in place

		if self.pretrained_CNN == 'VGG19':
			imgs = preprocess_input_vgg19(imgs)
		elif self.pretrained_CNN == 'InceptionV3':
			imgs_processed = []
			for iimg in range(imgs.shape[0]):
				imgs_processed.append(np.array(Image.fromarray(imgs[iimg].astype('uint8')).resize((299,299))))

			imgs_processed = np.array(imgs_processed)
			imgs = preprocess_input_inceptionv3(imgs_processed)
		elif self.pretrained_CNN == 'Densenet169':
			imgs = preprocess_input_densenet169(imgs)


		return self.model.predict(imgs).T






