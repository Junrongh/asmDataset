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

fc1=np.load('../data/preprocessed/asm-vgg16-fc1.npy')
fc1=np.load('../data/preprocessed/asm-vgg16-fc2.npy')


from sklearn.decomposition import PCA
pca_model = PCA(n_components=2)

x_pca = pca_model.fit_transform(X)
plt.figure(figsize=(6,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=class_ids)
plt.axis('equal')



