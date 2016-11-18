import numpy as np
import pickle
import sys, os
import os.path

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
K.set_image_dim_ordering('th')

if __name__== '__main__':
    data_path= sys.argv[1]
    out_model= sys.argv[2]
    lab_img= pickle.load(open(os.path.join(data_path, 'all_label.p'), 'rb')) #[0~9][0~499][3072]
    unlab_img= pickle.load(open(os.path.join(data_path, 'all_unlabel.p'), 'rb')) #[0~44999][3072]
    #data_path = os.path.join(os.getcwd(), 'data')
    #lab_img= pickle.load(open(os.path.join(data_path, 'all_label.p'), 'rb')) #[0~9][0~499][3072]
    #unlab_img= pickle.load(open(os.path.join(data_path, 'all_unlabel.p'), 'rb')) #[0~44999][3072]
                    
    img_rows, img_cols = 32, 32
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    batch_size= 128
    nb_epoch = 50
    
    ###initial train&test data###
    
    lab_img= np.array(lab_img)
    unlab_img= np.array(unlab_img)
    X_train= np.empty([4500, 3, 32, 32])
    Y_train= np.empty([4500, 10])
    X_validation= np.empty([500, 3, 32, 32])
    Y_validation= np.empty([500, 10])
    count1= 0
    count2= 0
    for cls in range(lab_img.shape[0]):
        for id in range(lab_img.shape[1]):
            if id < 450:
                X_train[count1] = lab_img[cls][id].reshape(3, 32, 32)
                target= np.zeros(10)
                target[cls]= 1
                Y_train[count1]= target
                #print(Y_train[count1])
                count1+= 1
            else:
                X_validation[count2] = lab_img[cls][id].reshape(3, 32, 32)
                target= np.zeros(10)
                target[cls]= 1
                Y_validation[count2]= target
                count2+= 1
                
    X_test= np.empty([45000, 3, 32, 32])
    Y_test= np.empty([45000, 10])
    for id in range(unlab_img.shape[0]):
        X_test[id]= unlab_img[id].reshape(3, 32, 32)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_validation = X_validation.astype('float32')
    X_train /= 255
    X_test /= 255
    X_validation /= 255
    
    np.save('X_train.npy', X_train)
    np.save('Y_train.npy', Y_train)
    np.save('X_test.npy', X_test)
    np.save('X_validation.npy', X_validation)
    np.save('Y_validation.npy', Y_validation)

    
    """  
    X_train= np.load('X_train.npy')
    Y_train= np.load('Y_train.npy')
    X_test= np.load('X_test.npy') 
    Y_validation= np.load('Y_validation.npy')
    X_validation= np.load('X_validation.npy')
    """
    datagen=  ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.2,#0.2
        height_shift_range=0.2,#0.2
        shear_range=0.1,#0.1
        zoom_range=0.1,#0.1
        horizontal_flip=True)#True
        #fill_mode='nearest')
    print('X_train shape:', X_train.shape)
    print('X_validation shape:', X_validation.shape)
    
    
    #######initial training########
    model = Sequential()

    model.add(Convolution2D(32, kernel_size[0], kernel_size[1], border_mode='same', input_shape=(3, 32, 32), dim_ordering='th' )) 
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1], dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, dim_ordering="th")) 
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, kernel_size[0], kernel_size[1], border_mode='same', dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, kernel_size[0], kernel_size[1], dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, dim_ordering="th"))
    model.add(Dropout(0.25))
    
    """ 
    model.add(Convolution2D(256, kernel_size[0], kernel_size[1], border_mode='same', dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, kernel_size[0], kernel_size[1], dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size, dim_ordering="th"))
    model.add(Dropout(0.25))
    """
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    adam= Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  #0.0005
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    checkpoint= ModelCheckpoint(out_model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    earlystop= EarlyStopping(monitor='val_acc', patience= 15, mode= 'max')
    datagen.fit(X_train)
    history= model.fit_generator(datagen.flow(X_train, Y_train, batch_size= batch_size, shuffle=True), samples_per_epoch= len(X_train)*10, nb_epoch=1,
                    verbose=1, validation_data= (X_validation, Y_validation), callbacks= [earlystop, checkpoint])
    del model
    model= load_model(out_model)
    score = model.evaluate(X_validation, Y_validation, verbose=0)
    print('Validation score:', score[0])
    print('Validation accuracy:', score[1])
    #model.save('test_model.h5')
    #del model
    
    
    #####semi-training####
    model=load_model(out_model)    
    for iteration in range(3):
        datagen.fit(X_train)
        checkpoint= ModelCheckpoint(out_model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        earlystop= EarlyStopping(monitor='val_acc', patience= 5, mode= 'max')

        model.fit_generator(datagen.flow(X_train, Y_train, batch_size= batch_size, shuffle=True), samples_per_epoch= len(X_train)*10, nb_epoch=15,
                    verbose=1, validation_data= (X_validation, Y_validation), callbacks= [earlystop, checkpoint])
        
        del model
        model= load_model(out_model)
        score = model.evaluate(X_validation, Y_validation, verbose=0)
        print('Validation score:', score[0])
        print('Validation accuracy:', score[1])
 
        Y_test= model.predict(X_test, batch_size=batch_size, verbose= 0)
        index= np.argwhere(Y_test>0.999)
        X_index= []
        Y_max= []
        for ii in index:
            X_index=np.append(X_index, ii[0])
            Y_max=np.append(Y_max, max(Y_test[ii[0]]))
        X_index= np.array(X_index, dtype=int)
        Y_max= np.array(Y_max, dtype=float)
        X_new= X_test[X_index]
        Y_new= np.equal(Y_test[X_index], Y_max.reshape(Y_max.shape[0], 1)).astype(float)
        
        X_new= np.concatenate((X_train, X_new), axis=0)
        Y_new= np.concatenate((Y_train, Y_new), axis=0)
        print (X_train.shape, X_new.shape)
        
        datagen.fit(X_new)
        checkpoint= ModelCheckpoint(out_model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        earlystop= EarlyStopping(monitor='val_acc', patience= 5, mode= 'max')

        model.fit_generator(datagen.flow(X_new, Y_new, batch_size= batch_size, shuffle=True), samples_per_epoch= len(X_new), nb_epoch=10,
                    verbose=1, validation_data= (X_validation, Y_validation), callbacks= [earlystop, checkpoint])
        
        del model
        model= load_model(out_model)
        score = model.evaluate(X_validation, Y_validation, verbose=0)
        print('Validation score:', score[0])
        print('Validation accuracy:', score[1])
        
    
    """
    #####atuoencoder#####
    input_img= Input(shape=(3,32,32))
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, output=encoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') 
    autoencoder.fit(X_train, X_train,
            nb_epoch=50,
            batch_size=128,
            shuffle=True,
            validation_data=(X_validation, X_validation))
    encoder.save('encoder.h5')
    encoded_train = encoder.predict(X_train)
    encoded_validation= encoder.predict(X_validation)
        
    for iteration in range(20):

        if os.path.isfile(out_model):
            model= load_model(out_model)
            print("found current_model !!")
        else:
            print(encoded_train.shape)
            model = Sequential()
            
            model.add(Reshape((256,), input_shape=(16,4,4)))
            model.add(Dense(256, input_shape=(256,)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            
            model.add(Dense(128))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            
            model.add(Dense(64))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            
            model.add(Dense(32))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(10))
            model.add(Activation('softmax'))
 
        adam= Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        checkpoint= ModelCheckpoint(out_model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        earlystop= EarlyStopping(monitor='val_acc', patience= 15, mode= 'max')
        
        model.fit(encoded_train, Y_train, batch_size= batch_size, shuffle=True, nb_epoch=50,verbose=1, 
                validation_data= (encoded_validation, Y_validation), callbacks= [earlystop, checkpoint])
        
        #print (history.history)
        del model
        model= load_model(out_model)
        score = model.evaluate(encoded_validation, Y_validation, verbose=0)
        print('Validation score:', score[0])
        print('Validation accuracy:', score[1])
        #model.save('test_model.h5')
        #del model
      
     
        encoded_test= encoder.predict(X_test)
        Y_test= model.predict(encoded_test, batch_size=batch_size, verbose= 0)
        index= np.argwhere(Y_test>0.95)
        X_index= []
        Y_max= []
        for ii in index:
            X_index=np.append(X_index, ii[0])
            Y_max=np.append(Y_max, max(Y_test[ii[0]]))
        X_index= np.array(X_index, dtype=int)
        Y_max= np.array(Y_max, dtype=float)
        X_new= encoded_test[X_index]
        Y_new= np.equal(Y_test[X_index], Y_max.reshape(Y_max.shape[0], 1)).astype(float)
        
        X_new= np.concatenate((encoded_train, X_new), axis=0)
        Y_new= np.concatenate((Y_train, Y_new), axis=0)
        print (encoded_train.shape, X_new.shape)
        
        checkpoint= ModelCheckpoint(out_model, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        earlystop= EarlyStopping(monitor='val_acc', patience= 15, mode= 'max')

        model.fit(X_new, Y_new, batch_size= batch_size, shuffle=True, nb_epoch=20,verbose=1, 
                validation_data= (encoded_validation, Y_validation), callbacks= [earlystop, checkpoint])
        
        del model
        model= load_model(out_model)
        score = model.evaluate(encoded_validation, Y_validation, verbose=0)
        print('Validation score:', score[0])
        print('Validation accuracy:', score[1])
    """  
    


     
    
