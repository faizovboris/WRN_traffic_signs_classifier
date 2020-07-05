import keras
import tensorflow as tf
import numpy as np

import keras.backend as K
from keras.models import load_model
from keras.engine import Input, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard,ReduceLROnPlateau
from keras.layers import add,merge, multiply, Dense, Activation, Flatten, Lambda, Conv2D, AveragePooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D


def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def subsample_and_shortcut(x, nb_filters=16, subsample_factor=1):
    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, data_format="channels_last")(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x

    if nb_filters > prev_nb_channels:
        shortcut = Lambda(zero_pad_channels,
                          arguments={'pad': nb_filters - prev_nb_channels})(shortcut)

    return subsample, shortcut


def residual_block(x, nb_filters=16, subsample_factor=1, dropout=0,
                   enable_squeeze_and_excite=False, SaE_reduction_rate=16):

    subsample, shortcut = subsample_and_shortcut(x, nb_filters, subsample_factor)

    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Conv2D(nb_filters, (3, 3),
               kernel_initializer="he_normal", padding="same", data_format="channels_last", strides=subsample)(y)

    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    if dropout:
        y = Dropout(dropout)(y)
    y = Conv2D(nb_filters, (3, 3),
        kernel_initializer="he_normal", padding="same", data_format="channels_last", strides=(1, 1))(y)

    if enable_squeeze_and_excite:
        s = GlobalAveragePooling2D()(y)
        height, width, num_channels = y.shape.as_list()[1:]
        s = Dense(num_channels//SaE_reduction_rate)(s)
        s = Activation('relu')(s)
        s = Dense(num_channels)(s)
        s = Activation('sigmoid')(s)
        s = Lambda(lambda x: K.reshape(x, (-1, 1, 1, num_channels)))(s)
        s = Lambda(lambda x: K.repeat_elements(x, height, axis=1))(s)
        s = Lambda(lambda x: K.repeat_elements(x, width, axis=2))(s)
        y = multiply([y, s])

    out = add([y, shortcut])

    return out


def input_layer(config):

    size = config['img']['img_size'] + 2 * config['img']['padding']
    img_rows, img_cols = size, size

    img_channels = 3
    inputs = Input(shape=(img_rows, img_cols, img_channels))

    return inputs


def conv1(inputs, config):
    x = Conv2D(16, (3, 3),
               padding="same", data_format="channels_last", kernel_initializer="he_normal")(inputs)
    return x


def conv2(x, config):
    blocks_per_group = config['wide_resnet']['metablock_depth']
    widening_factor = config['wide_resnet']['width']
    dropout = config['wide_resnet']['dropout']
    enable_squeeze_and_excite = config['wide_resnet'].get('enable_squeeze_and_excite', False)
    SaE_reduction_rate = config['wide_resnet'].get('SaE_reduction_rate', 16)

    for i in range(0, blocks_per_group):
        nb_filters = 16 * widening_factor
        x = residual_block(x, nb_filters=nb_filters,
                           subsample_factor=1, dropout=dropout,
                           enable_squeeze_and_excite=enable_squeeze_and_excite,
                           SaE_reduction_rate=SaE_reduction_rate)

    return x


def conv3(x, config):
    blocks_per_group = config['wide_resnet']['metablock_depth']
    widening_factor = config['wide_resnet']['width']
    enable_squeeze_and_excite = config['wide_resnet'].get('enable_squeeze_and_excite', False)
    SaE_reduction_rate = config['wide_resnet'].get('SaE_reduction_rate', 16)

    for i in range(0, blocks_per_group):
        nb_filters = 32 * widening_factor
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = residual_block(x, nb_filters=nb_filters,
                           subsample_factor=subsample_factor,
                           enable_squeeze_and_excite=enable_squeeze_and_excite,
                           SaE_reduction_rate=SaE_reduction_rate)

    return x


def conv4(x, config):
    blocks_per_group = config['wide_resnet']['metablock_depth']
    widening_factor = config['wide_resnet']['width']
    enable_squeeze_and_excite = config['wide_resnet'].get('enable_squeeze_and_excite', False)
    SaE_reduction_rate = config['wide_resnet'].get('SaE_reduction_rate', 16)

    for i in range(0, blocks_per_group):
        nb_filters = 64 * widening_factor
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = residual_block(x, nb_filters=nb_filters,
                           subsample_factor=subsample_factor,
                           enable_squeeze_and_excite=enable_squeeze_and_excite,
                           SaE_reduction_rate=SaE_reduction_rate)

    return x


def wresnet(config):

    inputs = input_layer(config)
    x = conv1(inputs, config)
    x = conv2(x, config)
    x = conv3(x, config)
    x = conv4(x, config)

    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid', data_format="channels_last")(x)
    x = Flatten()(x)

    predictions_features = Dense(1050, activation=None)(x)
    predictions_features = BatchNormalization()(predictions_features)
    predictions_features = Activation('relu')(predictions_features)
    predictions = Dense(config['wide_resnet']['num_classes'], activation='softmax')(predictions_features)

    model = Model(inputs=inputs, outputs=[predictions_features, predictions])
    return model

def contrastive_loss(y_true, y_pred):
    margin = 2.0
    y_pred = y_pred / K.expand_dims(tf.norm(y_pred, axis=-1) + 1e-6, 1)
    x_ = K.expand_dims(y_pred, 0)
    y_ = K.expand_dims(y_pred, 1)
    s = x_ - y_
    dist = K.sqrt(K.sum(K.square(s), axis=-1) + 1e-6)
    ys = K.dot(y_true[:, :205], K.transpose(y_true[:, :205]))
    #zs = 1 - K.dot(y_true[:, 205:206], K.transpose(y_true[:, 205:206]))
    #loss = 0.5 * K.sum(zs * ys * dist) / K.sum(ys * zs) + 0.5 * K.sum(zs * (1 - ys) * K.maximum(margin - dist, 0)) / K.sum((1 - ys) * zs)
    loss = 0.5 * K.sum(ys * K.square(dist)) / K.sum(ys) + 0.5 * K.sum((1 - ys) * K.square(K.maximum(margin - dist, 0))) / K.sum(1 - ys)
    return loss

def prepare_optimizer(config):
    from keras.optimizers import RMSprop
    lr = config['schedule']['lr']
    opt = Adam(lr=lr, decay=5e-4)
    return opt


def prepare_scheduler(config):

    def lr_sch(epoch):
        lr = config['schedule']['lr']
        gamma = config['schedule']['gamma']
        step = config['schedule']['lr']

        return lr * gamma**(epoch//step)

    return LearningRateScheduler(lr_sch)


def prepare_model(config):

    opt = prepare_optimizer(config)
    model = wresnet(config)
    model.compile(optimizer=opt,
                  loss=[contrastive_loss, 'categorical_crossentropy'],
                  metrics=['accuracy'])

    return model

def zero_loss(y_true, y_pred):
    return 0 * K.mean(y_pred)

def get_callbacks(config):
    lr_scheduler = prepare_scheduler(config)
    return [lr_scheduler]



