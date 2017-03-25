# -*- coding: utf-8 -*-

import numpy as np

seed = 7
np.random.seed(seed)

#值得注意的是keras的基层函数库是Theano还是TensorFlow#
#二者区别在于image的表现方式，前者将channel数放在最前，后者放在最后#

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

from matplotlib import pyplot as plt
from skimage.io import imread, imsave, imshow

#导入MNIST的手写数字图片数据#
#XTrain:(60000, 28, 28)，yTrain:(60000, )，XTest:(10000, 28, 28)，yTest:(10000, )#
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()

#显式声明图片的深度，MNIST灰度图片，深度为1，RGB的3通道全彩图片，深度为3#
XTrain = XTrain.reshape(XTrain.shape[0], 1, 28, 28)
XTest = XTest.reshape(XTest.shape[0], 1, 28, 28)

#原本labels为数字，将其改造为N×10的数组，y[n]中为1的列数i表示第n张图片的label是i-1#
yTrain = np_utils.to_categorical(yTrain, 10)
yTest = np_utils.to_categorical(yTest, 10)

#构建CNN模型#
model = Sequential()

#Convolution层，32个卷积过滤器，3×3卷积内核，model的第一层输入中应标明输入的数据类型input_shape#
model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(1, 28, 28)))

#MaxPooling层减少模型参数数量，使用2×2的滤波器，并从其中取最大值#
model.add(MaxPooling2D(pool_size=(2, 2)))

#Dropout层防止过度拟合#
model.add(Dropout(0.25))


model.add(Flatten())

#Dense层#
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#声明损失函数以及优化器#
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(XTrain, yTrain, batch_size=32, epochs=10, verbose=1)

score = model.evaluate(XTest, yTest, verbose=0)
