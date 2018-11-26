"""
Notes:
    + Window size in save/load mm's is hardcoded. Probably make this a param at some point.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


SPLITS_DIR = 'splits/'
D_TYPE = 'float32'


def save_training_results(history, experiment_results_dir, experiment_id, model):
    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.title(experiment_id + ' training loss')
    plt.savefig(os.path.join(experiment_results_dir, experiment_id + '_valloss.png'))

    # Model Architecture
    model_json = model.to_json()
    with open(os.path.join(experiment_results_dir, experiment_id + ".json"), "w") as json_file:
        json_file.write(model_json)

    # Model History
    with open(os.path.join(experiment_results_dir, experiment_id + "_hist"), 'wb') as file:
        pickle.dump(history.history, file)

    # Weights
    model.save_weights(os.path.join(experiment_results_dir, experiment_id + ".h5"))

    print "Training completed."
    print "Results in:\n"
    print experiment_results_dir


def create_split_dirs(dataset_id):
    """
    Creates specific preprocessed dataset directories.
    *refer to Notes about val dir.
    :param dataset_id: str - unique id of dataset_config, transform_type, and model.
    Returns
    -------
    dict - contains relevant experiment paths.
    """

    dataset_path = os.path.join(SPLITS_DIR, dataset_id)
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')
    expect_dir = os.path.join(dataset_path, 'expect')

    # Check for splits folder
    if not os.path.exists(SPLITS_DIR):
        os.mkdir(SPLITS_DIR)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(expect_dir):
        os.mkdir(expect_dir)

    paths = {}
    paths['train'] = train_dir
    paths['val'] = val_dir
    paths['test'] = test_dir
    paths['expect'] = expect_dir

    return paths


def create_adsr_expect_dirs(paths):
    new_paths = paths

    expect_yOns_dir = os.path.join(paths['expect'], 'yOns')
    expect_yFroms_dir = os.path.join(paths['expect'], 'yFroms')
    expect_yOffs_dir = os.path.join(paths['expect'], 'yOffs')

    if not os.path.exists(expect_yOns_dir):
        os.mkdir(expect_yOns_dir)
    if not os.path.exists(expect_yFroms_dir):
        os.mkdir(expect_yFroms_dir)
    if not os.path.exists(expect_yOffs_dir):
        os.mkdir(expect_yOffs_dir)

    new_paths['expect_yOns'] = expect_yOns_dir
    new_paths['expect_yFroms'] = expect_yFroms_dir
    new_paths['expect_yOffs'] = expect_yOffs_dir

    return new_paths

def save_mm(path, datapoint):
    mm_datapoint = np.memmap(
                        filename=path,
                        mode='w+',
                        dtype=D_TYPE,
                        shape=datapoint.shape)
    mm_datapoint[:] = datapoint[:]
    del mm_datapoint


def load_logfilt_mm(data_dir, type, ID):
    NOTE_RANGE = 88
    WINDOW_SIZE = 5
    N_BINS = 229

    input_path = os.path.join(data_dir, type, ID)
    output_path = os.path.join(data_dir, 'expect', ID)

    mm_input = np.memmap(input_path, mode='r', dtype=D_TYPE)
    mm_output = np.memmap(output_path, mode='r', dtype=D_TYPE)
    input = np.reshape(mm_input, (-1, WINDOW_SIZE, N_BINS))
    output = np.reshape(mm_output, (-1, NOTE_RANGE))

    return input, output


def load_logfilt_adsr_mm(data_dir, type, ID):
    NOTE_RANGE = 88
    WINDOW_SIZE = 11
    N_BINS = 144

    input_path = os.path.join(data_dir, type, ID)
    yOns_path = os.path.join(data_dir, 'expect/yOns/', ID)
    yFroms_path = os.path.join(data_dir, 'expect/yFroms/', ID)
    yOffs_path = os.path.join(data_dir, 'expect/yOffs/', ID)

    mm_input = np.memmap(input_path, mode='r', dtype=D_TYPE)
    mm_yOn = np.memmap(yOns_path, mode='r', dtype=D_TYPE)
    mm_yFrom = np.memmap(yFroms_path, mode='r', dtype=D_TYPE)
    mm_yOff = np.memmap(yOffs_path, mode='r', dtype=D_TYPE)

    input = np.reshape(mm_input, (-1, WINDOW_SIZE, N_BINS))
    yOn = np.reshape(mm_yOn, (-1, NOTE_RANGE))
    yFrom = np.reshape(mm_yFrom, (-1, NOTE_RANGE))
    yOff = np.reshape(mm_yOff, (-1, NOTE_RANGE))

    return input, yOn, yFrom, yOff


def load_logfilt_shallow_mm(data_dir, type, ID):
    NOTE_RANGE = 88
    N_BINS = 96
    # LogFilt
    # 48 = nb, 229
    # 36 = nb, 185
    # 24 = nb, 136
    # 12 = nb, 78

    # CQT
    # 48 = nb, 384
    # 36 = nb, 288
    # 24 = nb, 192
    # 12 =nb , 96

    input_path = os.path.join(data_dir, type, ID)
    output_path = os.path.join(data_dir, 'expect', ID)

    mm_input = np.memmap(input_path, mode='r', dtype=D_TYPE)
    mm_output = np.memmap(output_path, mode='r', dtype=D_TYPE)
    input = np.reshape(mm_input, (-1, N_BINS))
    output = np.reshape(mm_output, (-1, NOTE_RANGE))

    return input, output


def load_hcqt_mm(data_dir, type, ID):
    NOTE_RANGE = 88
    N_BINS = 360
    HARMONICS = 6
    WINDOW_SIZE = 5

    input_path = os.path.join(data_dir, type, ID)
    output_path = os.path.join(data_dir, 'expect', ID)

    mm_input = np.memmap(input_path, mode='r', dtype=D_TYPE)
    mm_output = np.memmap(output_path, mode='r', dtype=D_TYPE)
    input = np.reshape(mm_input, (-1, WINDOW_SIZE, N_BINS, HARMONICS))
    output = np.reshape(mm_output, (-1, NOTE_RANGE))

    return input, output


def load_hcqt_shallow_mm(data_dir, type, ID):
    NOTE_RANGE = 88

    N_BINS = 72
    # 48 numbands = 288, 60 numbands = 360
    # 36 nb = 216, 24 nb = 144
    # 12 nb = 72
    HARMONICS = 6

    input_path = os.path.join(data_dir, type, ID)
    output_path = os.path.join(data_dir, 'expect', ID)

    mm_input = np.memmap(input_path, mode='r', dtype=D_TYPE)
    mm_output = np.memmap(output_path, mode='r', dtype=D_TYPE)
    input = np.reshape(mm_input, (-1, N_BINS, HARMONICS))
    output = np.reshape(mm_output, (-1, NOTE_RANGE))

    return input, output


def fetch_config2_paths(config, args):
    """
    Fetches train and test sets based on Sigtia Configuration 2
    :param config:
    :return: np array, np array - train and test .wav paths
    """
    train_wav_paths = []
    test_wav_paths = []

    root_dir = ""
    if args.dataset_config == 'config-2':
        root_dir = config['DATASET_DIR']
    elif args.dataset_config == 'maps_subset_config2':
        root_dir = config['MAPS_SUBSET_CONFIG2']


    test_dirs = ""
    if args.dataset_config == 'config-2':
        test_dirs = config['DATASET_CONFIGS']['config-2']['test']
    elif args.dataset_config == 'maps_subset_config2':
        test_dirs = config['DATASET_CONFIGS']['maps_subset_config2']['test']


    for subdir_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue
        for dir_parent, _, file_names in os.walk(subdir_path):
            for name in file_names:
                if name.endswith('.wav'):
                    track_name = name.split('.wav')[0]
                    midi_name = track_name + '.mid'
                    if midi_name in file_names:
                        wav_path = os.path.join(dir_parent, name)
                        if subdir_name in test_dirs:
                            test_wav_paths.append(wav_path)
                        else:
                            train_wav_paths.append(wav_path)

    return train_wav_paths, test_wav_paths

def fetch_config2_fold_1_paths(config, args):
    root_dir = config['MAPS_CONFIG2_FOLD_1']
    test_dir = config['DATASET_CONFIGS']['maps_config2_fold_1_adsr']['test']

    train_wav_paths = []
    test_wav_paths = []

    for some_file in os.listdir(root_dir):
        if some_file.endswith('.wav'):
            wav_path = os.path.join(root_dir, some_file)
            train_wav_paths.append(wav_path)

    for some_file in os.listdir(test_dir):
        if some_file.endswith('.wav'):
            wav_path = os.path.join(root_dir, some_file)
            test_wav_paths.append(wav_path)

    return train_wav_paths, test_wav_paths


def fetch_config1_fold_3_paths(config, args):
    """
    Fetches train and test sets based on Sigtia Configuration 2
    :param config:
    :return: np array, np array - train and test .wav paths
    """
    train_wav_paths = []
    test_wav_paths = []

    root_dir = config['MAPS_CONFIG1_FOLD_3']
    test_dirs = config['DATASET_CONFIGS']['maps_config1_fold_3']['test']


    for subdir_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue
        for dir_parent, _, file_names in os.walk(subdir_path):
            for name in file_names:
                if name.endswith('.wav'):
                    track_name = name.split('.wav')[0]
                    midi_name = track_name + '.mid'
                    if midi_name in file_names:
                        wav_path = os.path.join(dir_parent, name)
                        if subdir_name in test_dirs:
                            test_wav_paths.append(wav_path)
                        else:
                            train_wav_paths.append(wav_path)

    return train_wav_paths, test_wav_paths
