import numpy as np
import pickle
import sys, os
import pandas as pd

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K

if __name__== '__main__':
    #data_path = os.path.join(os.getcwd(), 'data')
    data_path = sys.argv[1]
    test_img= pickle.load(open(os.path.join(data_path, 'test.p'), 'rb')) #[0~9][0~499][3072]
    
    X_test= np.empty([10000, 3, 32, 32])
    count= 0
    for test in test_img['data']:
        X_test[count]= np.array(test).reshape(3,32,32)
        count+= 1
    X_test= X_test.astype('float32')/ 255
    model= load_model(sys.argv[2])
    encoder= load_model('encoder.h5')
    X_test= encoder.predict(X_test)
    Y_test=model.predict(X_test, batch_size= 128, verbose= 0)
    Y_test= np.argmax(Y_test, axis= 1).astype(int)
    Y_test= np.hstack(('class', Y_test))
    id_list=['ID']
    for id in range(10000):
        id_list.append(id)
    table= pd.Series(Y_test, index= id_list)
    table.to_csv(sys.argv[3])
    

