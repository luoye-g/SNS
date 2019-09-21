from keras.models import *
import keras.layers as layer
import tensorflow as tf
from keras.layers import Conv2D , BatchNormalization , PReLU , MaxPooling2D , Dropout , UpSampling2D , concatenate


def mycrossentropy(y_true , y_pred , weight = [0.9 , 0.1]):

    k_pre = y_true * y_pred
    c_pre = (1 - y_true) * y_pred

    k_loss = K.mean(K.binary_crossentropy(k_pre , y_true) , axis = -1)
    c_loss = K.mean(K.binary_crossentropy(1 - c_pre , 1 - y_true) , axis = -1)

    return k_loss * weight[0] + c_loss * weight[1]

def iou(y_true , y_pred):
    return tf.count_nonzero(y_true * y_pred) / tf.count_nonzero(y_true)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def residual_block(x , channels):
    rx = Conv2D(channels , 3 , padding='same' , kernel_initializer='he_normal')(x)
    rx = BatchNormalization(axis = 3)(rx)
    rx = PReLU()(rx)

    x = Conv2D(channels , kernel_size=3 , strides=1 , padding='same')(x)
    x = BatchNormalization(axis = 3)(x)
    return layer.add([x , rx])

def bottom_layer(x , channels):
    x = Conv2D(channels, 3 , padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis = 3)(x)
    x = PReLU()(x)

    x = Conv2D(channels, 3 , padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis = 3)(x)
    x = PReLU()(x)

    return x

def unet_ori(input_shape = (256 , 256 , 3) , output_channel = 1):
    inputs = Input(input_shape)

    conv1 = Conv2D(64 , 3 , padding='same' , kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis = 3)(conv1)
    conv1 = PReLU()(conv1)

    down_layer1 = residual_block(conv1 , 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down_layer1)

    down_layer2 = residual_block(pool1 , 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down_layer2)

    down_layer3 = residual_block(pool2 , 256)
    pool3 = MaxPooling2D(pool_size=(2 , 2))(down_layer3)

    bottom = bottom_layer(pool3 , 512)

    up_sample3 = UpSampling2D(size = (2 , 2))(bottom)
    up_layer3 = residual_block(concatenate([down_layer3 , up_sample3] , axis = 3) , channels = 128)

    up_sample2 = UpSampling2D(size = (2 , 2))(up_layer3)
    up_layer2 = residual_block(concatenate([down_layer2, up_sample2], axis = 3), channels = 64)

    up_sample1 = UpSampling2D(size = (2 , 2))(up_layer2)
    up_layer1 = residual_block(concatenate([down_layer1, up_sample1], axis = 3) , channels = 64)

    output = Conv2D(3 , 3 , padding='same' , kernel_initializer='he_normal')(up_layer1)
    output = BatchNormalization(axis = 3)(output)
    output = PReLU()(output)

    output = Conv2D(output_channel, 1, activation='softmax')(output)

    model = Model(inputs = inputs, output = output)

    return model

def unet(input_shape = (256 , 256 , 3) , output_channel = 1):
    inputs = Input(input_shape)

    conv1 = Conv2D(64 , 3 , padding='same' , kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization(axis = 3)(conv1)
    conv1 = PReLU()(conv1)

    down_layer1 = residual_block(conv1 , 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down_layer1)

    down_layer2 = residual_block(pool1 , 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down_layer2)

    down_layer3 = residual_block(pool2 , 256)
    pool3 = MaxPooling2D(pool_size=(2 , 2))(down_layer3)

    down_layer4 = residual_block(pool3 , 512)
    pool4  =MaxPooling2D(pool_size = (2 , 2))(down_layer4)

    bottom = bottom_layer(pool4 , 512)

    up_sample4 = UpSampling2D(size = (2 , 2))(bottom)
    up_layer4 = residual_block(concatenate([down_layer4 , up_sample4] , axis = 3) , channels = 256)

    up_sample3 = UpSampling2D(size = (2 , 2))(up_layer4)
    up_layer3 = residual_block(concatenate([down_layer3 , up_sample3] , axis = 3) , channels = 128)

    up_sample2 = UpSampling2D(size = (2 , 2))(up_layer3)
    up_layer2 = residual_block(concatenate([down_layer2, up_sample2], axis = 3), channels = 64)

    up_sample1 = UpSampling2D(size = (2 , 2))(up_layer2)
    up_layer1 = residual_block(concatenate([down_layer1, up_sample1], axis = 3) , channels = 64)

    output = Conv2D(3 , 3 , padding='same' , kernel_initializer='he_normal')(up_layer1)
    output = BatchNormalization(axis = 3)(output)
    output = PReLU()(output)

    output = Conv2D(output_channel, 1, activation='softmax')(output)

    model = Model(inputs = inputs, output = output)

    return model

def unet_4down(input_shape = (256 , 256 , 3) , output_channel = 3):
    inputs = Input(input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = residual_block(conv1 , 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = residual_block(conv2 , 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = residual_block(conv3 , 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = residual_block(conv4 , 512)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = residual_block(conv6 , 512)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = residual_block(conv7 , 256)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = residual_block(conv8 , 128)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = residual_block(conv9 , 64)

    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(output_channel, 1, activation='softmax')(conv9)

    model = Model(input=inputs, output=conv10)

    return model