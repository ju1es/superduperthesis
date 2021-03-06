from keras.models import Model
from keras.layers import Input, Dense, Dropout, Input, Conv2D, Reshape, Activation
from keras.layers import ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.layers import GaussianDropout, GaussianNoise
from keras.layers.normalization import BatchNormalization


def shallow_net(input_shape):
    input = Input(input_shape)
    hidden = Dense(512, activation='relu')(input)
    output = Dense(88, activation='sigmoid')(hidden)

    return Model(inputs=input, outputs=output)


def hcqt_shallow_net(input_shape):
    input = Input(input_shape)
    flatten = Flatten()(input)
    hidden = Dense(512, activation='relu')(flatten)
    output = Dense(88, activation='sigmoid')(hidden)

    return Model(inputs=input, outputs=output)


def hcqt_conv(input_shape):
    inputs = Input(shape=input_shape)
    b1 = BatchNormalization()(inputs)
    c1 = Conv2D(32, (5, 5), padding='same', activation='relu')(b1)
    b2 = BatchNormalization()(c1)
    c2 = Conv2D(32, (5, 5), padding='same', activation='relu')(b2)
    b3 = BatchNormalization()(c2)
    c3 = Conv2D(32, (3, 3), padding='same', activation='relu')(b3)
    b4 = BatchNormalization()(c3)
    c4 = Conv2D(8, (3, 3), padding='same', activation='relu')(b4)
    b5 = BatchNormalization()(c4)
    f = Flatten()(b5)
    outputs = Dense(88, activation='sigmoid')(f)

    return Model(inputs=inputs, outputs=outputs)


def adsr_conv(input_shape=(11, 144)):
    # Init stack
    inputs = Input(shape=input_shape, name='main_input')
    reshape = Reshape((11, 144, 1))(inputs)
    c1 = Conv2D(30, (3, 3), activation='elu')(reshape)
    gd1 = GaussianDropout(0.1)(c1)
    gn1 = GaussianNoise(0.1)(gd1)

    c2 = Conv2D(30, (1, 24), activation='elu')(gn1)
    gd2 = GaussianDropout(0.1)(c2)
    gn2 = GaussianNoise(0.1)(gd2)

    c3 = Conv2D(30, (7, 1), activation='elu')(gn2)
    gd3 = GaussianDropout(0.1)(c3)
    gn3 = GaussianNoise(0.1)(gd3)

    # y-on
    c4 = Conv2D(10, (3, 3), activation='elu')(gn3)
    gd4 = GaussianDropout(0.5)(c4)
    gn4 = GaussianNoise(0.1)(gd4)
    f1 = Flatten()(gn4)
    yOn = Dense(88, activation='sigmoid', name='yOn')(f1)

    # y-frm
    c5 = Conv2D(10, (3, 3), activation='elu')(gn3)
    gd5 = GaussianDropout(0.5)(c5)
    gn5 = GaussianNoise(0.1)(gd5)
    f2 = Flatten()(gn5)
    yFrom = Dense(88, activation='sigmoid', name='yFrom')(f2)

    # y-off
    c6 = Conv2D(10, (3, 3), activation='elu')(gn3)
    gd6 = GaussianDropout(0.5)(c6)
    gn6 = GaussianNoise(0.1)(gd6)
    f3 = Flatten()(gn6)
    yOff = Dense(88, activation='sigmoid', name='yOff')(f3)

    return Model(inputs=inputs, outputs=[yOn, yFrom, yOff])


def hcqt_adsr_conv(input_shape):
    # HCQT dims
    inputs = Input(shape=input_shape, name='main_input')

    c1 = Conv2D(30, (3, 5), padding='same', activation='elu')(inputs)
    gd1 = GaussianDropout(0.1)(c1)
    gn1 = GaussianNoise(0.1)(gd1)

    c2 = Conv2D(30, (1, 36), activation='elu')(gn1)
    gd2 = GaussianDropout(0.1)(c2)
    gn2 = GaussianNoise(0.1)(gd2)

    c3 = Conv2D(30, (3, 1), activation='elu', padding='same')(gn2)
    gd3 = GaussianDropout(0.1)(c3)
    gn3 = GaussianNoise(0.1)(gd3)

    # y-on
    c4 = Conv2D(10, (3, 3), activation='elu')(gn3)
    gd4 = GaussianDropout(0.5)(c4)
    gn4 = GaussianNoise(0.1)(gd4)
    f1 = Flatten()(gn4)
    yOn = Dense(88, activation='sigmoid', name='yOn')(f1)

    # y-frm
    c5 = Conv2D(10, (3, 3), activation='elu')(gn3)
    gd5 = GaussianDropout(0.5)(c5)
    gn5 = GaussianNoise(0.1)(gd5)
    f2 = Flatten()(gn5)
    yFrom = Dense(88, activation='sigmoid', name='yFrom')(f2)

    # y-off
    c6 = Conv2D(10, (3, 3), activation='elu')(gn3)
    gd6 = GaussianDropout(0.5)(c6)
    gn6 = GaussianNoise(0.1)(gd6)
    f3 = Flatten()(gn6)
    yOff = Dense(88, activation='sigmoid', name='yOff')(f3)

    return Model(inputs=inputs, outputs=[yOn, yFrom, yOff])



def baseline_cnn(input_shape, window_size):
    # OLD
    # Layers
    input = Input(input_shape)
    reshape = Reshape((5, 229, 1))(input)

    conv_1 = Conv2D(32, (3, 3), activation='relu')(reshape)
    zero_pad = ZeroPadding2D(padding=(window_size // 2, window_size // 2))(conv_1)

    # Add linear conv layer
    conv_2 = Conv2D(32, (3, 3))(zero_pad)
    # Apply batch norm
    batch_norm = BatchNormalization()(conv_2)
    # Then activation layer
    nonlin_act = Activation('relu')(batch_norm)

    max_pool_1 = MaxPooling2D(pool_size=(1, 2))(nonlin_act)
    dropout_1 = Dropout(0.25)(max_pool_1)

    conv_3 = Conv2D(64, (3, 3), activation='relu')(dropout_1)

    max_pool_2 = MaxPooling2D(pool_size=(1, 2))(conv_3)

    flatten = Flatten()(max_pool_2)
    dropout_2 = Dropout(0.25)(flatten)
    dense_1 = Dense(512, activation='relu')(dropout_2)
    dropout_3 = Dropout(0.5)(dense_1)
    output = Dense(88, activation='sigmoid')(dropout_3)

## NEW
    # input = Input(input_shape)
    # reshape = Reshape((5, 229, 1))(input)
    #
    # conv_1 = Conv2D(32, (3, 3), activation='relu')(reshape)
    # zero_pad = ZeroPadding2D(padding=(window_size // 2, window_size // 2))(conv_1)
    # batch_norm_1 = BatchNormalization()(zero_pad)
    # conv_2 = Conv2D(32, (3, 3), activation='relu')(batch_norm_1)
    # batch_norm_2 = BatchNormalization()(conv_2)
    # max_pool_1 = MaxPooling2D(pool_size=(1, 2))(batch_norm_2)
    # dropout_1 = Dropout(0.25)(max_pool_1)
    #
    # conv_3 = Conv2D(64, (3, 3), activation='relu')(dropout_1)
    # batch_norm_3 = BatchNormalization()(conv_3)
    # max_pool_2 = MaxPooling2D(pool_size=(1, 2))(batch_norm_3)
    # flatten = Flatten()(max_pool_2)
    # dropout_2 = Dropout(0.25)(flatten)
    #
    # dense_1 = Dense(512, activation='relu')(dropout_2)
    # batch_norm_4 = BatchNormalization()(dense_1)
    # dropout_3 = Dropout(0.5)(batch_norm_4)
    #
    # output = Dense(88, activation='sigmoid')(dropout_3)

    return Model(inputs=input, outputs=output)


def baseline_dcnn(inputShape):
    input = Input(inputShape)

    conv1 = Conv2D(32, (3, 3), activation='relu')(input)
    conv2 = Conv2D(32, (3, 3), activations='relu')(conv1)
    batchNorm1 = BatchNormalization()(conv2)
    maxPool1 = MaxPooling2D(pool_size=(1, 2))(batchNorm1)
    dropOut1 = Dropout(0.25)(maxPool1)
    conv3 = Conv2D(32, (3, 3), activations='relu')(dropOut1)
    batchNorm2 = BatchNormalization()(conv3)
    conv4 = Conv2D(32, (3, 3), activations='relu')(batchNorm2)
    batchNorm3 = BatchNormalization()(conv4)
    maxPool2 = MaxPooling2D(pool_size=(1, 2))(batchNorm3)
    dropOut2 = Dropout(0.25)(maxPool2)
    conv5 = Conv2D(64, (1, 25), activations='relu')(dropOut2)
    batchNorm4 = BatchNormalization()(conv5)
    conv6 = Conv2D(128, (1, 25), activations='relu')(batchNorm4)
    batchNorm5 = BatchNormalization()(conv6)
    dropOut3 = Dropout(0.5)(batchNorm5)
    conv7 = Conv2D(88, (1, 1), activations='relu')(dropOut3)
    batchNorm6 = BatchNormalization()(conv7)
    avgPool1 = AveragePooling2D(pool_size=(1, 6))(batchNorm6)
    output = Dense(88, activation='sigmoid')(avgPool1)

    return Model(inputs=input, outputs=output)
