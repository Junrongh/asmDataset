<<<<<<< HEAD
# Keras: Deep Learning library for TensorFlow and Theano
=======
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

	
	
>>>>>>> origin/master

[![Build Status](https://travis-ci.org/fchollet/keras.svg?branch=master)](https://travis-ci.org/fchollet/keras)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fchollet/keras/blob/master/LICENSE)

## You have just found Keras.

Keras is a high-level neural networks API, written in Python and capable of running on top of either [TensorFlow](https://github.com/tensorflow/tensorflow) or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. *Being able to go from idea to result with the least possible delay is key to doing good research.*

Use Keras if you need a deep learning library that:

- Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
- Supports both convolutional networks and recurrent networks, as well as combinations of the two.
- Runs seamlessly on CPU and GPU.

Read the documentation at [Keras.io](http://keras.io).

Keras is compatible with: __Python 2.7-3.5__.


------------------


## Guiding principles

- __User friendliness.__ Keras is an API designed for human beings, not machines. It puts user experience front and center. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.

- __Modularity.__ A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.

- __Easy extensibility.__ New modules are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Keras suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.


------------------


## Getting started: 30 seconds to Keras

The core data structure of Keras is a __model__, a way to organize layers. The simplest type of model is the [`Sequential`](http://keras.io/getting-started/sequential-model-guide) model, a linear stack of layers. For more complex architectures, you should use the [Keras functional API](http://keras.io/getting-started/functional-api-guide), which allows to build arbitrary graphs of layers.

Here is the `Sequential` model:

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. A core principle of Keras is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).
```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:

```python
model.train_on_batch(x_batch, y_batch)
```

Evaluate your performance in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

Building a question answering system, an image classification model, a Neural Turing Machine, or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

- [Getting started with the Sequential model](http://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](http://keras.io/getting-started/functional-api-guide)

In the [examples folder](https://github.com/fchollet/keras/tree/master/examples) of the repository, you will find more advanced models: question-answering with memory networks, text generation with stacked LSTMs, etc.


------------------


## Installation

Keras uses the following dependencies:

- numpy, scipy
- yaml
- HDF5 and h5py (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.


*When using the TensorFlow backend:*

- TensorFlow
    - [See installation instructions](https://www.tensorflow.org/install/).

*When using the Theano backend:*

- Theano
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

To install Keras, `cd` to the Keras folder and run the install command:
```sh
sudo python setup.py install
```

You can also install Keras from PyPI:
```sh
sudo pip install keras
```

------------------


## Switching from TensorFlow to Theano

By default, Keras will use TensorFlow as its tensor manipulation library. [Follow these instructions](http://keras.io/backend/) to configure the Keras backend.

------------------


## Support

You can ask questions and join the development discussion:

- On the [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).
- On the [Keras Slack channel](https://kerasteam.slack.com). Use [this link](https://keras-slack-autojoin.herokuapp.com/) to request an invitation to the channel.

You can also post **bug reports and feature requests** (only) in [Github issues](https://github.com/fchollet/keras/issues). Make sure to read [our guidelines](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md) first.


------------------


## Why this name, Keras?

Keras (κέρας) means _horn_ in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the _Odyssey_, where dream spirits (_Oneiroi_, singular _Oneiros_) are divided between those who deceive men with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------
