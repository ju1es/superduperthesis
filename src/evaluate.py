import os
import numpy as np
import json
from keras.models import model_from_json
from keras.optimizers import SGD

RESULTS_DIR = './results/'
SPLITS_DIR = './splits/'

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


def eval_framewise(predicts, targets, thresh=0.5):
    """
     author: filip (+ data-format amendments by rainer)
     """
    if predicts.shape != targets.shape:
         raise ValueError('predictions.shape {} != targets.shape {}'.format(predictions.shape, targets.shape))

    pred = predicts > thresh
    targ = targets > thresh

    tp = pred & targ
    fp = pred ^ tp
    fn = targ ^ tp

    # tp, fp, tn, fn
    return tp.sum(), fp.sum(), 0, fn.sum()


def run(config, args, experiment_id):
    '''
    Generates metrics for a given model.
    :param config:
    :param args:
    :param id:
    :return: Generates precision, recall, and f-measure of a given model
    '''

    model = []
    MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, experiment_id)
    MODEL_SPLIT_DIR = os.path.join(SPLITS_DIR, experiment_id)
    if args.model == 'baseline':
        # Load model
        json_file = open(os.path.join(MODEL_RESULTS_DIR, experiment_id + '.json'))
        loaded_baseline_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_baseline_json)
        model.load_weights(os.path.join(MODEL_RESULTS_DIR, experiment_id + '.h5'))
        model.compile(
                loss='binary_crossentropy',
                optimizer=SGD(lr=0.1, momentum=0.9),
                metrics=['accuracy', 'mse', 'mae'])

    if args.dataset_config == 'config-2_subset':
        # Load test set
        datapoints_path = os.path.join(SPLITS_DIR, experiment_id, 'test')
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

    tp_total, fp_total, fn_total = 0, 0, 0
    for p, t in zip(predictions, y):
        tp, fp, _, fn = eval_framewise(p, t)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / float(tp_total + fp_total)
    recall = tp_total / (tp_total + float(fn_total))
    accuracy = tp_total / float(tp_total + fp_total + fn_total)
    f_measure = (2 * precision * recall) / float(precision + recall)

    print '\n\n totals: tp, fp, fn'
    print tp_total, fp_total, fn_total
    print '\n precision, recall, accuracy, f_measure'
    print precision, recall, accuracy, f_measure

    # Save
    results_file_path = os.path.join(MODEL_RESULTS_DIR, 'results.txt')
    with open(results_file_path, 'w') as results_file:
        results_file.write("precision recall f_measure accuracy\n")
        results_file.write(str(precision) + " " + str(recall) + " " + str(f_measure) + " " + str(accuracy))

    print "Saved results at " + results_file_path



