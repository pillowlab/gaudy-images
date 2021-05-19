
import os
from os import walk
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

from tensorflow.keras import layers



class ResnetFeaturesClass:

 
### INIT FUNCTION

	def __init__(self):

		# load ResNet model
		resnet_model = load_model('/usr/people/bcowley/adroit/code/synthAL/saved_models/resnet_model.h5')
		x = AveragePooling2D(pool_size=(2,2), padding='valid')(resnet_model.get_layer('activation_27').output)
		self.resnet_model = Model(inputs=resnet_model.input, outputs=x)


### CALL FUNCTIONS

	def get_features_from_imgs(self, imgs_raw):
		# assume imgs is in resnet form (bgr)

		imgs = np.copy(imgs_raw)
		imgs = preprocess_input_resnet50(imgs)
		return self.resnet_model.predict(imgs)












