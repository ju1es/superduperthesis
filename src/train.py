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
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD

SPLITS_DIR = 'splits/'
RESULTS_DIR = 'results/'
VAL_PERCENTAGE = 0.18

def run(config, args, dataset_id, experiment_id):
    '''
    Trains a model based on specifications
    :param config: dict - config.
    :param args: namespace - args passed in.
    :param dataset_id: str - id.
    :param experiment_id: str - id.
    :return:
    '''

    # Check if preprocess has been performed
    if not os.path.exists(os.path.join(SPLITS_DIR, dataset_id)):
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
    elif args.model == 'shallow_net':
        model = m.shallow_net(input_shape=(96,))
    elif args.model == 'hcqt_shallow_net':
        model = m.hcqt_shallow_net(input_shape=(72, 6))
    elif args.model == 'adsr_net_baseline':
        model = m.adsr_conv(input_shape=(11, 144))
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
        dataset_dir = os.path.join(SPLITS_DIR, dataset_id)
        train_datapoints = os.listdir(os.path.join(dataset_dir, 'train'))

        # Load datapoints for .fit()
        X, y = [], []
        for dat_file in train_datapoints:
            input, output = wrangler.load_logfilt_mm(dataset_dir, 'train', dat_file)

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
        wrangler.save_training_results(history, experiment_results_dir, experiment_id, model)

    elif args.model == 'adsr_net_baseline':
        config = config['MODELS']['baseline']['TRAIN']
        # Compile
        model.compile(
            loss={'yOn':'binary_crossentropy', 'yFrom':'binary_crossentropy', 'yOff':'binary_crossentropy'},
            optimizer=SGD(lr=config['LR'], momentum=config['MOMENTUM'], nesterov=True),
            metrics=['accuracy', 'mse', 'mae'])

        # Fetch train wav paths
        dataset_dir = os.path.join(SPLITS_DIR, dataset_id)
        train_datapoints = os.listdir(os.path.join(dataset_dir, 'train'))

        # Load datapoints for .fit()
        X, yOns, yFroms, yOffs = [], [], [], []
        for dat_file in train_datapoints:
            input, yOn, yFrom, yOff = wrangler.load_logfilt_adsr_mm(dataset_dir, 'train', dat_file)

            X.append(input)
            yOns.append(yOn)
            yFroms.append(yFrom)
            yOffs.append(yOff)

        X = np.concatenate(X)
        yOns = np.concatenate(yOns)
        yFroms = np.concatenate(yFroms)
        yOffs = np.concatenate(yOffs)


        # Callbacks
        csv_logger = CSVLogger(RESULTS_DIR + experiment_id + "/"+ experiment_id + ".log")
        checkpoint = ModelCheckpoint(
                    RESULTS_DIR + experiment_id + "/"+ experiment_id + "_checkpoint.h5",
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min')
        early_stopping = EarlyStopping(patience=12, monitor='val_loss', verbose=1, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
        history = model.fit(
                    X,
                    {"yOn": yOns, "yFrom": yFroms, "yOff": yOffs},
                    epochs=config['EPOCHS'],
                    batch_size=config['BATCH_SIZE'], # 8 according to Rainer ICASSP18
                    callbacks=[reduce_lr, checkpoint, early_stopping, csv_logger],
                    validation_split=VAL_PERCENTAGE,
                    verbose=1)

        # Save
        # -> acc, val_acc plot
        # -> loss, val_loss plot
        # -> model history
        # -> weights
        wrangler.save_training_results(history, experiment_results_dir, experiment_id, model)

    # Train
    elif args.model == 'shallow_net':
        config = config['MODELS']['shallow_net']['TRAIN']
        # Compile
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=config['LR'], momentum=config['MOMENTUM']),
            metrics=['accuracy', 'mse', 'mae'])

        # Fetch train wav paths
        dataset_dir = os.path.join(SPLITS_DIR, dataset_id)
        train_datapoints = os.listdir(os.path.join(dataset_dir, 'train'))

        # Load datapoints for .fit()
        X, y = [], []
        for dat_file in train_datapoints:
            input, output = wrangler.load_logfilt_shallow_mm(dataset_dir, 'train', dat_file)

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
        wrangler.save_training_results(history, experiment_results_dir, experiment_id, model)


    elif args.model == 'hcqt-conv':
        config = config['MODELS']['hcqt-conv']['TRAIN']
        # Compile
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=config['LR'], momentum=config['MOMENTUM']),
            metrics=['accuracy', 'mse', 'mae'])

        # Fetch train wav paths
        dataset_dir = os.path.join(SPLITS_DIR, dataset_id)
        train_datapoints = os.listdir(os.path.join(dataset_dir, 'train'))

        # Load datapoints for .fit()
        X, y = [], []
        for dat_file in train_datapoints:
            input, output = wrangler.load_hcqt_mm(dataset_dir, 'train', dat_file)

            X.append(input)
            y.append(output)

        X = np.concatenate(X)
        y = np.concatenate(y)

        # Callbacks
        decay = HalfDecay(config['LR'], config['HALVING_N_EPOCHS'])
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
            batch_size=config['BATCH_SIZE'],
            callbacks=[decay, checkpoint, early_stopping, csv_logger],
            validation_split=VAL_PERCENTAGE,
            verbose=1)

        # Save
        # -> acc, val_acc plot
        # -> loss, val_loss plot
        # -> model history
        # -> weights
        wrangler.save_training_results(history, experiment_results_dir, experiment_id, model)
    elif args.model == 'hcqt_shallow_net':
        config = config['MODELS']['hcqt_shallow_net']['TRAIN']
        # Compile
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=config['LR'], momentum=config['MOMENTUM']),
            metrics=['accuracy', 'mse', 'mae'])

        # Fetch train wav paths
        dataset_dir = os.path.join(SPLITS_DIR, dataset_id)
        train_datapoints = os.listdir(os.path.join(dataset_dir, 'train'))

        # Load datapoints for .fit()
        X, y = [], []
        for dat_file in train_datapoints:
            input, output = wrangler.load_hcqt_shallow_mm(dataset_dir, 'train', dat_file)

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
        wrangler.save_training_results(history, experiment_results_dir, experiment_id, model)
