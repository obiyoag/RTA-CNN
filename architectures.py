import tensorflow as tf
from keras import  backend as K
from keras.models import Model, load_model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers import Reshape,multiply
from keras.layers.merge import add, concatenate
from keras.layers import Dropout
from keras.layers import Conv1D, BatchNormalization,LSTM
from keras.layers import ZeroPadding1D, UpSampling1D,Cropping1D
from keras.layers.pooling import MaxPooling1D


__all__ = ['RTA_CNN', 'VGG12', 'RESNET50', 'MSCNN', 'ATICNN', '1DCNN']

# RTA-CNN
def conv_block(in_x, nb_filter, kernel_size):

    x = Conv1D(nb_filter, kernel_size, padding='same')(in_x) 
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)       
    return x

def attention_branch(in_x, nb_filter, kernel_size):
    
    x1 = conv_block(in_x, nb_filter, kernel_size) 

    x = MaxPooling1D(2)(x1)
    x = conv_block(x, nb_filter, kernel_size)  
    x = UpSampling1D(size = 2)(x)

    x2 = conv_block(x, nb_filter, kernel_size)    
    
    if(K.int_shape(x1)!=K.int_shape(x2)):
        x2 = ZeroPadding1D(1)(x2)
        x2 = Cropping1D((1,0))(x2)

    x = add([x1, x2])    

    x = conv_block(x, nb_filter, kernel_size)

    x = Conv1D(nb_filter, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    
    return x

def RTA_block(in_x, nb_filter, kernel_size):
    
    x1 = conv_block(in_x, nb_filter, kernel_size)
    x2 = conv_block(x1, nb_filter, kernel_size)
    
    attention_map = attention_branch(x1, nb_filter, kernel_size)
    
    x = multiply([x2, attention_map])
    x = add([x, x1])
    
    out = conv_block(x, nb_filter, kernel_size)
    
    return out

def RTA_CNN():
    
    inputs = Input((9000, 1))
    
    x = RTA_block(inputs, 16, 32)
    x = MaxPooling1D(4)(x)
    
    x = RTA_block(x, 32, 16)
    x = MaxPooling1D(4)(x)
    
    x = RTA_block(x, 64, 9)
    x = MaxPooling1D(2)(x)
    x = RTA_block(x, 64, 9)
    x = MaxPooling1D(2)(x)
    
    x = RTA_block(x, 128, 3)
    x = MaxPooling1D(2)(x)
    x = RTA_block(x, 128, 3)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(100,  activation='relu')(x)
    x = Dense(3,  activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model


#1DCNN
def WDCNN():
    
    inputs = Input((9000, 1))
    
    x = Conv1D(16, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(100,  activation='relu')(x)
    x = Dense(3,  activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model


# VGG12
def VGG12():
    
    inputs = Input((9000, 1))
    
    x = Conv1D(64, 3, padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)

    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)

    x = Conv1D(512, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(512, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(512, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)
    

    x = Flatten()(x)
    x = Dense(100,  activation='relu')(x)
    x = Dense(3,  activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model

# RESNET50
def identity_block(in_x, nb_filters):

    F1, F2, F3 = nb_filters
    
    x = in_x
    
    x = Conv1D(F1, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv1D(F2, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(F3, 1,  padding="same")(x)
    x = BatchNormalization()(x)
    
    x = add([x, in_x])
    x = Activation("relu")(x)
    
    return x

def convolutional_block(in_x, nb_filters, stride):

    F1, F2, F3 = nb_filters
    
    x = in_x

    x = Conv1D(F1, 1, strides=stride,  padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(F2, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(F3, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x1 = Conv1D(F3, 1, strides=stride, padding='same')(in_x)
    x1 = BatchNormalization()(x1)

    x = add([x, x1])
    x = Activation('relu')(x)
    
    return x

def RESNET50():
    
    inputs = Input((9000, 1))
    
    x = Conv1D(64, 7, strides=2)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(3, stride=2)(x)

    x = convolutional_block(x, [64, 64, 256], 4)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])
    
    x = convolutional_block(x, [128, 128, 512], 4)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    
    x = convolutional_block(x, [256, 256, 1024], 2)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    
    x = convolutional_block(x, [512, 512, 2048], 2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(500, activation="relu")(x)
    x = Dense(3, activation="softmax")(x)
    
    model = Model(inputs, x)

    return model

def MSCNN():
    
    inputs = Input((9000, 1))
    
    x1 = Conv1D(64, 3, padding='same')(inputs)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(64, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=3)(x1)
    
    x1 = Conv1D(128, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(128, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=3)(x1)
    
    x1 = Conv1D(256, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(256, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(256, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=2)(x1)
    
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=2)(x1)
    
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=2)(x1)
    
    
    
    x2 = Conv1D(64, 3, padding='same')(inputs)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(64, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=3)(x2)
    
    x2 = Conv1D(128, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(128, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=3)(x2)
    
    x2 = Conv1D(256, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(256, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(256, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=2)(x2)
    
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=2)(x2)
    
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=2)(x2)
    
    x = concatenate([x1 , x2] , axis=-1)
    
    
    x = Flatten()(x)
    x = Dense(1024,  activation='relu')(x)
    x = Dense(1024,  activation='relu')(x)
    x = Dense(256,  activation='relu')(x)
    x = Dense(3,  activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model


#ATICNN
def ATICNN():
    
    inputs = Input((9000, 1))
    
    x = Conv1D(64, 3, padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = LSTM(256,return_sequences=True)(x)
    x = LSTM(256,return_sequences=True)(x)
    
    x = GlobalAveragePooling1D()(x)
    x1 = Dense(256, activation="relu")(x)
    x1 = Dense(256, activation="softmax")(x1)
    x1 = multiply([x1, x])
    x = add([x1, x])
    x = Dense(3, activation="softmax")(x)
    
    model = Model(inputs, x)
    
    return model


def focal_loss(y_true, y_pred):
    
    epsilon = 1.e-7
    alpha = tf.constant([[1],[1],[1]], dtype=tf.float32)
    gamma = float(0.3)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    pos_pred = tf.pow(-tf.log(y_pred),gamma)
    nag_pred = tf.pow(-tf.log(1-y_pred),gamma)
    y_t = tf.multiply(y_true, pos_pred) + tf.multiply(1-y_true, nag_pred)
    loss = tf.matmul(y_t, alpha)
    loss = tf.reduce_mean(loss)
    return loss