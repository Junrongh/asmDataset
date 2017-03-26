#		Test&Learning Packages

##		Scikit-mage

tool box for scipy，import for the image reading, editing, saving, etc.

Useful Command:
imread, imsave, imshow
>
	import skimage
	import skimage.io
	from skimage.io import imread, imsave, imshow
>
	#import matplotlib.pyplot for showing images#
	import matplotlib.pyplot as plt
>	
	#'path', str, full path for a image, including extension '.png', etc.#
	image=imread('path')
>	
	show=imshow(image)
	plt.show(show)
>	
	#'path', str, same as introduced above#
	imsave('path', image)

skimage.color
>
	from skimage.io import imread, imsave, imshow
	import skimage.color
	from skimage.color import gray2rgb, rgb2gray
	image=imread('path')
>	
	#gray type image is a 2d-array，rgb type image is a 3d-array#
	#Using gray2rgb is turning a 2d-array to a 3d-array#
	#by adding a layer of 0#
	image_gray=rgb2gray(image)
	image_rgb=gray2rgb(image)
	
	
##		Keras

tool box for Deep Learning, building CNN architecture

Unseful Command: 
Model, Sequential, Dense, Dropout, Activation, Flatten, Convolution2D, Maxpooling2D, etc. 

>
	import keras
>	
	#Find the backend package in use#
	>>>Using Theano backend
	or
	>>>Using TensorFlow backend

The main difference between Theano package and TensorFlow package in the Keras case is that :

Theano's image input is a 3d-array, shape=(Channel, x-pixels, y-pixels)

TensorFlow's image input is a 3d-array, shape=(x-pixels, y-pixels, Channel)

>
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Convolution2D, MaxPooling2D
	from keras.utils import np_utils
	
>
	#Building CNN model#
	model=Sequential()
	model.add(layer)
>
	#layers can be Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, etc.#
>	
	#Convolution2D:#
	#N: Number of convolutional filter#
	#(x, y): shape of convolutional kernel#
	#deep: deep of a image, 1 for gray, 3 for RGB#
	#(shapex, shapey): shape of input image#
	#*Theano's input_shape=(deep, shapex, shapey)*#
	#*TensorFlow's input_shape=(shapex, shapey, deep)*#
	layer=Convolution2D(N, (x, y), padding='same', input_shape=(deep, shapex, shapey))
>
	#Maxpooling2D#
	#(x, y): the shape of filter#
	layer=MaxPooling2D(pool_size=(x, y))

>
	#Dropout, for fixing overfitting#
	#value: #
	layer=Dropout(value)
>
	#Flatten#
	layer=Flatten()
>
	#Dense#
	#value:#
	#activation='type', including relu, softmax, etc.#
	layer=Dense(value, activation='type')
>
	model.compile(loss='categorical_crossentropy',
			    optimizer='adam', metrics=['accuracy'])
	model.fit(XTrain, yTrain, batch_size=32, epochs=10, verbose=1)
	score=model.evaluate(XTest, yTest, verbose=0)

	
	

