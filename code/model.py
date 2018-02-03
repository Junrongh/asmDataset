import os
import numpy as np
import pandas as pd
import sqlite3

from scipy.misc import imresize
from skimage.io import imread
from skimage.color import gray2rgb

import keras.backend as K
from keras.models import Model
from keras.applications import vgg16
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers

conn = sqlite3.connect('../data/info.sqlite')
cursor = conn.cursor()
sql = "select * from info"

df = pd.read_sql(sql, conn)
N = len(df)
X = np.zeros([N, 224, 224])
y = np.zeros([N, ])

dic_main = {}
collection = ['al', 'as', 'cc', 'ci', 'co', 'cs', 'cu', 'hs', 'lz', 'mg', 'ni',
              'pl', 'rf', 'sc', 'sp', 'ss', 'ti', 'ts', 'un']
for i in range(0, 19):
    dic_main[collection[i]]=i
for i in range(0, N):
    path = df.loc[i]['scaled_image'].encode('utf-8').decode('utf-8')
    X[i, :, :] = imread(path, as_grey=True)
    main = df.loc[i]['main'].encode('utf-8').decode('utf-8')
    y[i] = dic_main[main]


def preprocess_imagenet(images):

    I = np.array([imresize(image, (224, 224)) for image in images])
    I = gray2rgb(I).astype(np.float32)

    return vgg16.preprocess_input(I)

def vgg_layer(output_name='fc1'):
    # Note: currently hippolyta compute nodes cannot access user home directories,
    # and do not have direct internet access.
    # use full paths to NFS filesystem endpoints (not symlinks)
    # keras pretrained model weight files are here:
    KERAS_ROOT = '/mnt/data/users/holmlab/.keras'
    weights_file = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = os.path.join(KERAS_ROOT, 'models', weights_file)

    # initialize network with random weights, and load from hdf5
    cnn = vgg16.VGG16(include_top=True, weights=None)
    cnn.load_weights(filepath=weights_path)

    model = Model(
        inputs=cnn.input,
        outputs=cnn.get_layer(output_name).output
    )
    return model


def prefeatures(images, layername='fc1', datadir='../data'):
        
    datafile = os.path.join(datadir, 'asm-vgg16-{}.npy'.format(layername))
    try:
        features = np.load(datafile)
    except FileNotFoundError:
        print('forward pass for all asm images')
        model = vgg_layer(layername)
        features = model.predict(preprocess_imagenet(images), verbose=True)
        np.save(datafile, features)

    return features

datadir='../data/preprocessed/'
fc1_all=prefeatures(X, layername='fc1', datadir=datadir)
fc2_all=prefeatures(X, layername='fc2', datadir=datadir)


