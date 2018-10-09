import os
import numpy as np
import keras

def read_mm(data_dir, type, ID):
    NOTE_RANGE = 88
    WINDOW_SIZE = 5
    N_BINS = 229

    input_path = os.path.join(data_dir, type, ID)
    output_path = os.path.join(data_dir, 'expect', ID)

    mm_input = np.memmap(input_path, mode='r')
    mm_output = np.memmap(output_path, mode='r')
    input = np.reshape(mm_input, (-1, WINDOW_SIZE, N_BINS))
    output = np.reshape(mm_output, (-1, NOTE_RANGE))

    return input, output


class DataGenerator(keras.utils.Sequence):
    def __init__(self, model_data_dir, data_type, list_IDs, batch_size=32, shuffle=True):
        self.model_data_dir = model_data_dir  # Denotes specific train/val/test for a given model
        self.data_type = data_type            # Denotes train, val, or test
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size datapoints'
        # Init
        X, y = [], []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            input, output = read_mm(self.model_data_dir, self.data_type, ID)

            X.append(input)
            y.append(output)

        # Flatten
        X = np.concatenate(X)
        y = np.concatenate(y)

        return X, y

    def __len__(self):
        'Denotes # of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[i] for i in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y