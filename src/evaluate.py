import os
import numpy as np
import json
from lib import models as m
from keras.models import model_from_json
from keras.optimizers import SGD
from lib import data_wrangler as wrangler

RESULTS_DIR = './results/'
SPLITS_DIR = './splits/'

def read_mm(data_dir, type, ID):
    NOTE_RANGE = 88
    WINDOW_SIZE = 5
    N_BINS = 229

    input_path = os.path.join(data_dir, type, ID)
    output_path = os.path.join(data_dir, 'expect', ID)

    mm_input = np.memmap(input_path, mode='r', dtype='float16')
    mm_output = np.memmap(output_path, mode='r', dtype='float16')
    input = np.reshape(mm_input, (-1, WINDOW_SIZE, N_BINS))
    output = np.reshape(mm_output, (-1, NOTE_RANGE))

    return input, output


def eval_framewise(predicts, targets, thresh=0.5):
    """
     author: filip (+ data-format amendments by rainer)
     """
    if predicts.shape != targets.shape:
         raise ValueError('predictions.shape {} != targets.shape {}'.format(predicts.shape, targets.shape))

    pred = predicts > thresh
    targ = targets > thresh

    tp = pred & targ
    fp = pred ^ tp
    fn = targ ^ tp

    # tp, fp, tn, fn
    return tp.sum(), fp.sum(), 0, fn.sum()


def prf_framewise((tp, fp, tn, fn)):
    tp, fp, tn, fn = float(tp), float(fp), float(tn), float(fn)

    if tp + fp == 0.:
        p = 0.
    else:
        p = tp / (tp + fp)

    if tp + fn == 0.:
        r = 0.
    else:
        r = tp / (tp + fn)

    if p + r == 0.:
        f = 0.
    else:
        f = 2 * ((p * r) / (p + r))

    if tp + fp + fn == 0.:
        a = 0.
    else:
        a = tp / (tp + fp + fn)

    return p, r, f, a


def run(config, args, dataset_id, experiment_id):
    '''
    Generates metrics for a given model.
    :param config:
    :param args:
    :param id:
    :return: Generates precision, recall, and f-measure of a given model
    '''

    model = []
    MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, experiment_id)
    MODEL_SPLIT_DIR = os.path.join(SPLITS_DIR, dataset_id)
    if args.model == 'baseline':
        # Load model
        # json_file = open(os.path.join(MODEL_RESULTS_DIR, experiment_id + '.json'))
        # loaded_baseline_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_baseline_json)
        model = m.baseline_cnn(input_shape=(5, 229), window_size=5)
        model.load_weights(os.path.join(MODEL_RESULTS_DIR, experiment_id + '.h5'))
        model.compile(
                loss='binary_crossentropy',
                optimizer=SGD(lr=0.1, momentum=0.9),
                metrics=['accuracy', 'mse', 'mae'])
    elif args.model == 'shallow_net':
        model = m.shallow_net(input_shape=(229,))
        weights = os.path.join(MODEL_RESULTS_DIR, experiment_id + '.h5')
        if not os.path.isfile(weights):
            weights = os.path.join(MODEL_RESULTS_DIR, experiment_id + '_checkpoint.h5')
        model.load_weights(weights)
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=0.1, momentum=0.9),
            metrics=['accuracy', 'mse', 'mae'])
    elif args.model == 'hcqt_shallow_net':
        model = m.hcqt_shallow_net(input_shape=(144,6)) # 60 numbands = 360, 60
        weights = os.path.join(MODEL_RESULTS_DIR, experiment_id + '.h5')
        if not os.path.isfile(weights):
            weights = os.path.join(MODEL_RESULTS_DIR, experiment_id + '_checkpoint.h5')
        model.load_weights(weights)
        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(lr=0.1, momentum=0.9),
            metrics=['accuracy', 'mse', 'mae'])

    if dataset_id == 'config-2_logfilt':
        # Load test set
        datapoints_path = os.path.join(SPLITS_DIR, dataset_id, 'test')
        test_datapoints = os.listdir(datapoints_path)

        X, y = [], []
        for dat_file in test_datapoints:
            input, output = read_mm(MODEL_SPLIT_DIR, 'test', dat_file)

            X.append(input)
            y.append(output)

        X = np.concatenate(X)
        y = np.concatenate(y)

        # Evaluate
        predictions = model.predict(X, verbose=1)

        # tp_total, fp_total, fn_total = 0, 0, 0
        # for p, t in zip(predictions, y):
        #     tp, fp, _, fn = eval_framewise(p, t)
        #     tp_total += tp
        #     fp_total += fp
        #     fn_total += fn
        #
        # precision = tp_total / float(tp_total + fp_total)
        # recall = tp_total / (tp_total + float(fn_total))
        # accuracy = tp_total / float(tp_total + fp_total + fn_total)
        # f_measure = (2 * precision * recall) / float(precision + recall)

        precision, recall, f_measure, accuracy = prf_framewise(eval_framewise(predictions, y, thresh=0.5))


        # print '\n\n totals: tp, fp, fn'
        # print tp_total, fp_total, fn_total
        print '\n precision, recall, accuracy, f_measure'
        print precision, recall, accuracy, f_measure

        # Save
        results_file_path = os.path.join(MODEL_RESULTS_DIR, 'results.txt')
        with open(results_file_path, 'w') as results_file:
            results_file.write("precision recall f_measure accuracy\n")
            results_file.write(str(precision) + " " + str(recall) + " " + str(f_measure) + " " + str(accuracy))

        print "Saved results at " + results_file_path

    elif dataset_id == 'config-2_logfilt_shallow' or dataset_id == 'maps_subset_config2_logfilt_shallow':
        datapoints_path = os.path.join(SPLITS_DIR, dataset_id, 'test')
        test_datapoints = os.listdir(datapoints_path)

        X, y = [], []
        for dat_file in test_datapoints:
            input, output = wrangler.load_logfilt_shallow_mm(MODEL_SPLIT_DIR, 'test', dat_file)

            X.append(input)
            y.append(output)

        X = np.concatenate(X)
        y = np.concatenate(y)

        # Evaluate
        predictions = model.predict(X, verbose=1)

        precision, recall, f_measure, accuracy = prf_framewise(eval_framewise(predictions, y, thresh=0.5))

        # print '\n\n totals: tp, fp, fn'
        # print tp_total, fp_total, fn_total
        print '\n precision, recall, accuracy, f_measure'
        print precision, recall, accuracy, f_measure

        # Save
        results_file_path = os.path.join(MODEL_RESULTS_DIR, 'results.txt')
        with open(results_file_path, 'w') as results_file:
            results_file.write("precision recall f_measure accuracy\n")
            results_file.write(str(precision) + " " + str(recall) + " " + str(f_measure) + " " + str(accuracy))

        print "Saved results at " + results_file_path

    elif dataset_id == 'config-2_hcqt_shallow' or dataset_id == 'maps_subset_config2_hcqt_shallow':
        datapoints_path = os.path.join(SPLITS_DIR, dataset_id, 'test')
        test_datapoints = os.listdir(datapoints_path)

        X, y = [], []
        for dat_file in test_datapoints:
            input, output = wrangler.load_hcqt_shallow_mm(MODEL_SPLIT_DIR, 'test', dat_file)

            X.append(input)
            y.append(output)

        X = np.concatenate(X)
        y = np.concatenate(y)

        # Evaluate
        predictions = model.predict(X, verbose=1)

        precision, recall, f_measure, accuracy = prf_framewise(eval_framewise(predictions, y, thresh=0.5))

        # print '\n\n totals: tp, fp, fn'
        # print tp_total, fp_total, fn_total
        print '\n precision, recall, accuracy, f_measure'
        print precision, recall, accuracy, f_measure

        # Save
        results_file_path = os.path.join(MODEL_RESULTS_DIR, 'results.txt')
        with open(results_file_path, 'w') as results_file:
            results_file.write("precision recall f_measure accuracy\n")
            results_file.write(str(precision) + " " + str(recall) + " " + str(f_measure) + " " + str(accuracy))

        print "Saved results at " + results_file_path