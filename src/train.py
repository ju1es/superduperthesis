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

from lib import data_wrangler as wrangler
from lib import errors as e
from lib import models as m
from lib.HalfDecay import HalfDecay
from lib.DataGenerator import DataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD

SPLITS_DIR = 'splits/'
RESULTS_DIR = 'results/'
VAL_PERCENTAGE = 0.18

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
    model = 0
    if args.model == 'baseline':
        model = m.baseline_cnn(input_shape=(5, 229), window_size=5)
    elif args.model == 'hcqt-conv':
        model = m.hcqt_conv(input_shape=(5, 360, 6))
    else:
        print "ERROR: MODEL DOESN\'T EXIST!"
        sys.exit()

    # Train
    if args.model == 'baseline':
        config = config['MODELS']['baseline']['TRAIN']
        # Compile
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=config['LR'], momentum=config['MOMENTUM']),
            metrics=['accuracy', 'mse', 'mae'])

        # Fetch train wav paths
        experiment_dir = os.path.join(SPLITS_DIR, experiment_id)
        train_datapoints = os.listdir(os.path.join(experiment_dir, 'train'))

        # Load datapoints for .fit()
        X, y = [], []
        for dat_file in train_datapoints:
            input, output = wrangler.load_logfilt_mm(experiment_dir, 'train', dat_file)

            X.append(input)
            y.append(output)

        X = np.concatenate(X)
        y = np.concatenate(y)


        # Callbacks
        decay = HalfDecay(config['LR'], config['HALVING_N_EPOCHS']) # 10 halving according to Rainer ICASSP18
        csv_logger = CSVLogger(RESULTS_DIR + experiment_id + "/"+ experiment_id + ".log")
        checkpoint = ModelCheckpoint(
                    RESULTS_DIR + experiment_id + "/"+ experiment_id + "_checkpoint.h5",
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min')
        early_stopping = EarlyStopping(patience=5, monitor='val_loss', verbose=1, mode='min')

        history = model.fit(
                    x=X,
                    y=y,
                    epochs=config['EPOCHS'],
                    batch_size=config['BATCH_SIZE'], # 8 according to Rainer ICASSP18
                    callbacks=[decay, checkpoint, early_stopping, csv_logger],
                    validation_split=VAL_PERCENTAGE,
                    verbose=1)

        # Save
        # -> acc, val_acc plot
        # -> loss, val_loss plot
        # -> model history
        # -> weights
        save_training_results(history, experiment_results_dir, experiment_id)


    elif args.model == 'hcqt-conv':
        config = config['MODELS']['hcqt-conv']['TRAIN']
        # Compile
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=config['LR'], momentum=config['MOMENTUM']),
            metrics=['accuracy', 'mse', 'mae'])

        # Fetch train wav paths
        experiment_dir = os.path.join(SPLITS_DIR, experiment_id)
        train_datapoints = os.listdir(os.path.join(experiment_dir, 'train'))

        # Load datapoints for .fit()
        X, y = [], []
        for dat_file in train_datapoints:
            input, output = wrangler.load_logfilt_mm(experiment_dir, 'train', dat_file)

            X.append(input)
            y.append(output)

        X = np.concatenate(X)
        y = np.concatenate(y)

        # Callbacks
        decay = HalfDecay(config['LR'], config['HALVING_N_EPOCHS'])  # 10 halving according to Rainer ICASSP18
        csv_logger = CSVLogger(RESULTS_DIR + experiment_id + "/" + experiment_id + ".log")
        checkpoint = ModelCheckpoint(
            RESULTS_DIR + experiment_id + "/" + experiment_id + "_checkpoint.h5",
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min')
        early_stopping = EarlyStopping(patience=5, monitor='val_loss', verbose=1, mode='min')

        history = model.fit(
            x=X,
            y=y,
            epochs=config['EPOCHS'],
            batch_size=config['BATCH_SIZE'],  # 8 according to Rainer ICASSP18
            callbacks=[decay, checkpoint, early_stopping, csv_logger],
            validation_split=VAL_PERCENTAGE,
            verbose=1)

        # Save
        # -> acc, val_acc plot
        # -> loss, val_loss plot
        # -> model history
        # -> weights
        save_training_results(history, experiment_results_dir, experiment_id)
