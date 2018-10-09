'''
Notes:
    + Hardcoded baseline args.
    + Hardcoded epochs.
'''
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

from lib import errors as e
from lib import models as m
from lib.HalfDecay import HalfDecay
from lib.DataGenerator import DataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

SPLITS_DIR = 'splits/'
RESULTS_DIR = 'results/'

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
        partitions = {}
        partitions['train'] = []
        partitions['val'] = []
        train_datapoints = os.listdir(os.path.join(SPLITS_DIR, experiment_id, 'train'))
        np.random.shuffle(train_datapoints)
        n_datapoints = len(train_datapoints)
        split_index = n_datapoints - int(0.18 * n_datapoints)
        partitions['train'], partitions['val'] = train_datapoints[:split_index], train_datapoints[split_index:]

        # Execute
        decay = HalfDecay(0.1, 5)
        checkpoint = ModelCheckpoint(
                    RESULTS_DIR + experiment_id + "/"+ experiment_id + "_checkpoint.h5",
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min')
        root_dir = os.path.join(SPLITS_DIR, experiment_id)
        train_gen = DataGenerator(root_dir, 'train', partitions['train'])
        val_gen = DataGenerator(root_dir, 'train', partitions['val'])
        history = model.fit_generator(
                    generator=train_gen,
                    validation_data=val_gen,
                    epochs=150, # Hardcoded
                    use_multiprocessing=True,
                    workers=6,
                    verbose=1,
                    callbacks=[decay, checkpoint])
        # Save
        # -> acc, val_acc plot
        # -> loss, val_loss plot
        # -> model history
        # -> weights
        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)

        experiment_results_dir = os.path.join(RESULTS_DIR, experiment_id)
        if not os.path.exists(experiment_results_dir):
            os.mkdir(experiment_results_dir)

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
