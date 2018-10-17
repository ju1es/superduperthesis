'''
Notes:
    + Hardcoded baseline args.
    + Hardcoded epochs.
'''
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pickle

from lib import errors as e
from lib import models as m
from lib.HalfDecay import HalfDecay
from lib.DataGenerator import DataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD

SPLITS_DIR = 'splits/'
RESULTS_DIR = 'results/'
VAL_PERCENTAGE = 0.18


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


def run(config, args, experiment_id):
    '''
    Trains a model based on specifications
    :param config: dict - config.
    :param args: namespace - args passed in.
    :param experiment_id: str - id.
    :return:
    '''

    # Check if preprocess has been performed
    if not os.path.exists(os.path.join(SPLITS_DIR, experiment_id)):
        e.print_no_data()
        sys.exit()

    # Setup results dir
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    experiment_results_dir = os.path.join(RESULTS_DIR, experiment_id)
    if not os.path.exists(experiment_results_dir):
        os.mkdir(experiment_results_dir)

    # Fetch model
    if args.model == 'baseline':
        model = m.baseline_cnn(input_shape=(5, 229), window_size=5)
    else:
        print "ERROR: MODEL DOESN\'T EXIST!"
        sys.exit()

    # Train
    if args.model == 'baseline':

        # Compile
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=0.1, momentum=0.9),
            metrics=['accuracy', 'mse', 'mae'])

        # Load and Shuffle IDs
        # partitions = {}
        # partitions['train'] = []
        # partitions['val'] = []
        experiment_dir = os.path.join(SPLITS_DIR, experiment_id)
        train_datapoints = os.listdir(os.path.join(experiment_dir, 'train'))
        # np.random.shuffle(train_datapoints)

        # # For .fit_generator()
        # n_datapoints = len(train_datapoints)
        # split_index = n_datapoints - int(VAL_PERCENTAGE * n_datapoints)
        # partitions['train'], partitions['val'] = train_datapoints[:split_index], train_datapoints[split_index:]

        # Load datapoints for .fit()
        X, y = [], []
        for dat_file in train_datapoints:
            input, output = read_mm(experiment_dir, 'train', dat_file)

            X.append(input)
            y.append(output)

        X = np.concatenate(X)
        y = np.concatenate(y)


        # Execute
        decay = HalfDecay(0.1, 10) # 10 halving according to Rainer ICASSP18
        checkpoint = ModelCheckpoint(
                    RESULTS_DIR + experiment_id + "/"+ experiment_id + "_checkpoint.h5",
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min')
        # early_stopping = EarlyStopping(patience=10, monitor='val_loss', verbose=1, mode='min')

        # # For .fit_generator()
        # root_dir = os.path.join(SPLITS_DIR, experiment_id)
        # train_gen = DataGenerator(root_dir, 'train', partitions['train'])
        # val_gen = DataGenerator(root_dir, 'train', partitions['val'])


        # # Use batches
        # history = model.fit_generator(
        #             generator=train_gen,
        #             validation_data=val_gen,
        #             epochs=150, # Hardcoded
        #             use_multiprocessing=True,
        #             workers=6,
        #             verbose=1,
        #             callbacks=[decay, checkpoint])

        history = model.fit(
                    x=X,
                    y=y,
                    epochs=50,
                    batch_size=8, # 8 according to Rainer ICASSP18
                    callbacks=[decay, checkpoint],
                    validation_split=VAL_PERCENTAGE,
                    verbose=1)

        # Save
        # -> acc, val_acc plot
        # -> loss, val_loss plot
        # -> model history
        # -> weights
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'val_loss'])
        plt.title(experiment_id + ' training loss')
        plt.savefig(os.path.join(experiment_results_dir, experiment_id + '_valloss.png'))

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train_acc', 'val_acc'])
        plt.title(experiment_id + ' training acc')
        plt.savefig(os.path.join(experiment_results_dir, experiment_id + '_acc.png'))

        model_json = model.to_json()
        with open(os.path.join(experiment_results_dir, experiment_id + ".json"), "w") as json_file:
            json_file.write(model_json)

        with open(os.path.join(experiment_results_dir, experiment_id + "_hist"), 'wb') as file:
            pickle.dump(history.history, file)

        model.save_weights(os.path.join(experiment_results_dir, experiment_id + ".h5"))

        print "Training completed."
        print "Results in:\n"
        print experiment_results_dir
