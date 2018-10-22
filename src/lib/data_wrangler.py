import os
import numpy as np

SPLITS_DIR = 'splits/'
D_TYPE = 'float16'

def create_split_dirs(dataset_id):
    '''
    Creates specific preprocessed dataset directories.
    *refer to Notes about val dir.
    :param dataset_id: str - unique id of dataset_config, transform_type, and model.
    Returns
    -------
    dict - contains relevant experiment paths.
    '''

    dataset_path = os.path.join(SPLITS_DIR, dataset_id)
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')
    expect_dir = os.path.join(dataset_path, 'expect')

    # Check for splits folder
    if not os.path.exists(SPLITS_DIR):
        os.mkdir(SPLITS_DIR)
    if not os.path.exists(dataset_path):
        os.mkdir(experiment_path)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(expect_dir):
        os.mkdir(expect_dir)

def save_mm(path, datapoint):
    '''
    Save datapoint to specified path
    :param path: str - path.
    :param datapoint: np array - datapoint.
    '''
    mm_datapoint = np.memmap(
                        filename=path,
                        mode='w+',
                        dtype=D_TYPE,
                        shape=datapoint.shape)
    mm_datapoint[:] = datapoint[:]
    del mm_datapoint


def fetch_config2_paths(config):
    '''
    Fetches train and test sets based on Sigtia Configuration 2
    :param config:
    :return: np array, np array - train and test .wav paths
    '''
    train_wav_paths = []
    test_wav_paths = []

    for subdir_name in os.listdir(config['DATASET_DIR']):
        subdir_path = os.path.join(config['DATASET_DIR'], subdir_name)
        if not os.path.isdir(subdir_path):
            continue
        for dir_parent, _, file_names in os.walk(subdir_path):
            for name in file_names:
                if name.endswith('.wav'):
                    track_name = name.split('.wav')[0]
                    midi_name = track_name + '.mid'
                    if midi_name in file_names:
                        wav_path = os.path.join(dir_parent, name)
                        test_dirs = config['DATASET_CONFIGS']['config-2']['test']
                        if subdir_name in test_dirs:
                            test_wav_paths.append(wav_path)
                        else:
                            train_wav_paths.append(wav_path)

    return train_wav_paths, test_wav_paths