from keras.models import Model
from keras.layers import Input, Dense, Dropout, Input, Conv2D, Reshape, Activation
from keras.layers import ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization

def baseline_cnn(input_shape, window_size):
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
    output = Dense(88, activation='relu')(dropout_3)

    return Model(inputs=input, outputs=output)


def baseline_dcnn(inputShape):
    input = Input(inputShape)

    conv1 = Conv2D(32, (3, 3), activation='relu')(input)
    conv2 = Conv2D(32, (3, 3), activations='relu')(conv2)
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
