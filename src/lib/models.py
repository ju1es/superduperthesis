from keras.models import Model
from keras.layers import Dense, Dropout, Input, Conv2D, Maxpooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

def baseline_cnn(inputShape):
    input = Input(inputShape)
    # reshape?
    # --> 

    #
    # 32 filters
    # 3x3
    #

    conv1 = Conv2D(32, (3, 3), activation='relu')(input)
    conv2 = Conv2D(32, (3, 3), activations='relu')(conv2)
    batchNorm = BatchNormalization()(conv2)
    maxPool1 = MaxPooling2D(pool_size=(1, 2))(batchNorm)
    dropOut1 = Dropout(0.25)(maxPool1)
    conv3 = Conv2D(64, (3, 3), activation='relu')(dropOut1)
    maxPool2 = MaxPooling2D(pool_size=(1, 2))(conv3)
    dropOut2 = Dropout(0.25)(maxPool2)
    dense1 = Dense(512, activation='relu')(dropOut2)
    dropOut3 = Dropout(0.5)(dense1)
    output = Dense(88, activation='relu')(dropOut3)

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
