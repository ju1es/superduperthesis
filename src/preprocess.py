'''
Notes:
    + For now, validation data lives in the train directories.
      Validation set is created during training from the train directory.
    + Hop_length in ground truth generation is hardcoded.
'''
import os
import madmom as mm
import librosa as lr
import pretty_midi as pm
import numpy as np
from lib import data_wrangler as wrangler

SPLITS_DIR = 'splits/'
NUM_DAT_FILES = 5

# Hack for PPQ from MAPS
pm.pretty_midi.MAX_TICK = 1e10

def _create_dest_dirs(experiment_id):
    '''
    Creates model's specific train and test directories.
    *refer to Notes about val dir.
    :param experiment_id: str - unique id of dataset_config, transform_type, and model.
    Returns
    -------
    dict - contains relevant experiment paths.
    '''

    experiment_path = os.path.join(SPLITS_DIR, experiment_id)
    train_dir = os.path.join(experiment_path, 'train')
    val_dir = os.path.join(experiment_path, 'val')
    test_dir = os.path.join(experiment_path, 'test')
    expect_dir = os.path.join(experiment_path, 'expect')
    results_path = os.path

    # Check for splits folder
    if not os.path.exists(SPLITS_DIR):
        os.mkdir(SPLITS_DIR)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(expect_dir):
        os.mkdir(expect_dir)

    experiment_paths = {}
    experiment_paths['experiment_path'] = experiment_path
    experiment_paths['train_dir'] = train_dir
    experiment_paths['val_dir'] = val_dir
    experiment_paths['test_dir'] = test_dir
    experiment_paths['expect_dir'] = expect_dir
    return experiment_paths


def _logfilt(config, track_path):
    '''
    Applies logarithmic filterbank stft with normalizing, rescaling, and windows.
    :param config: dict - config.
    :param track_path: str - path of track to transform.
    :return: np array - transformed track.
    '''
    log_spect = mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
        track_path,
        num_bands=config['NUM_BANDS'],
        num_channels=config['NUM_CHANNELS'],
        sr=config['SR'],
        unique_filters=config['UNIQUE_FILTERS'],
        norm_filters=config['NORM_FILTERS'],
        frame_size=config['FRAME_SIZE'],
        fft_size=config['FFT_SIZE'],
        fmin=config['FMIN'],
        fmax=config['FMAX'],
        fref=config['FREF'],
        circular_shift=config['CIRC_SHIFT'],
        fps=config['FPS'])

    # Normalize, Rescale
    log_spect = np.nan_to_num(log_spect)
    log_spect = np.abs(log_spect)
    log_spect /= np.max(log_spect, axis=0)

    # Generate Windows:
    log_spect = np.pad(
        log_spect,
        ((config['WINDOW_SIZE'] // 2, config['WINDOW_SIZE'] // 2), (0, 0)),
        'constant')

    windows = []
    for i in range(log_spect.shape[0] - config['WINDOW_SIZE'] + 1):
        w = log_spect[i:i + config['WINDOW_SIZE'], :]
        windows.append(w)

    return np.array(windows)


def _generate_expected(config, midi_path, input_shape, sr):
    '''
    Generates expected array off of associative midi.
    :param config: config.
    :param midi_path: path to midi generate expected off of.
    :param input_shape: shape of input.
    :param sr: sample rate.
    :return: np array - ground truth.
    '''
    pm_midi = pm.PrettyMIDI(midi_path)
    times = lr.frames_to_time(
                np.arange(input_shape),
                sr=sr,
                hop_length=441.0)
    expected = pm_midi.get_piano_roll(
                        fs=sr,
                        times=times)[config['MIN_MIDI']:config['MAX_MIDI']+1].T
    expected[expected > 0] = 1
    return expected


def _transform_track(config, args, track_path):
    '''
    Applies transform based on args.
    :param config: dict - config.
    :param args: namespace - passed in during execution.
    :param track_path: str - track to transform.
    :return: np array - Transformed track.
    '''
    X = []
    if args.transform_type == 'logfilt':
        X = _logfilt(config['TRANSFORMS']['logfilt'], track_path)

    return X


def _get_sample_rate(transform_config, args):
    '''
    Gets sample rate depending on transform type.
    :param transform_config: dict - transforms specs.
    :param args: namespace - passed in during execution.
    :return: int - sample rate.
    '''
    sr = 0
    if args.transform_type == 'logfilt':
        sr = transform_config['logfilt']['SR']

    return sr


def _preprocess_config2(config, args, paths, id):
    '''
    Generates processed .wav's and ground truths in experiment directory.
    :param config: dict - config.
    :param args: namespace - passed in during execution.
    :param paths: dict - train, val, test directories for experiment.
    :param id: str - unique id of experiment.
    '''
    # for subdir_name in os.listdir(config['DATASET_DIR']):
    #     subdir_path = os.path.join(config['DATASET_DIR'], subdir_name)
    #
    #     # Check if directory and not file.
    #     if not os.path.isdir(subdir_path):
    #         continue
    #
    #     # Find all .wavs
    #     for dir_parent, _, file_names in os.walk(subdir_path):
    #         for name in file_names:
    #             if name.endswith('.wav'):
    #                 track_name = name.split('.wav')[0]
    #                 midi_name = track_name + '.mid'
    #
    #                 if midi_name in file_names:
    #                     track_path = os.path.join(dir_parent, name)
    #                     midi_path = os.path.join(dir_parent, midi_name)
    #
    #             ### DELETE ###
    #                     print "Processing " + track_path
    #
    #                     # Transform and Generate ground truth
    #                     sr = _get_sample_rate(config['TRANSFORMS'], args)
    #                     np_input = _transform_track(config, args, track_path)
    #                     np_output = _generate_expected(config, midi_path, np_input.shape[0], sr)
    #
    #             ### DELETE ###
    #                     print np_input.shape
    #                     print np_output.shape
    #
    #                     ## Key component to Sigtia Configuration 2 ##
    #                     # -> train on synthetic, test on accoustic
    #                     datapoint_id = track_name + '.dat'
    #                     input_path = paths['train_dir']
    #                     test_dirs = config['DATASET_CONFIGS']['config-2']['test']
    #                     if subdir_name in test_dirs:
    #                         input_path = paths['test_dir']
    #
    #                     # Save transform and ground truth
    #                     input_path = os.path.join(input_path, datapoint_id)
    #                     output_path = os.path.join(paths['expect_dir'], datapoint_id)
    #                     wrangler.save_mm(input_path, np_input)
    #                     wrangler.save_mm(output_path, np_output)

    # Get all .wav paths
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

    # Shuffle
    np.random.shuffle(train_wav_paths)
    np.random.shuffle(test_wav_paths)

    # Preprocess train .wavs and save into X dat files.

## DELETE ##
    print "\nProcessing Training Files.\n"
    cur_dat_num = 0
    train_wav_paths = np.array_split(np.array(train_wav_paths), NUM_DAT_FILES)
    for dat_file in train_wav_paths:
        inputs, outputs = [], []
        for wav_path in dat_file:
            midi_path = wav_path.split('.wav')[0] + '.mid'

        ### DELETE ###
            print "Processing " + wav_path

            sr = _get_sample_rate(config['TRANSFORMS'], args)
            np_input = _transform_track(config, args, wav_path)
            np_output = _generate_expected(config, midi_path, np_input.shape[0], sr)

        ### DELETE ###
            print np_input.shape
            print np_output.shape

            inputs.append(np_input)
            outputs.append(np_output)

        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)

        input_path = os.path.join(paths['train_dir'], str(cur_dat_num) + '.dat')
        output_path = os.path.join(paths['expect_dir'], str(cur_dat_num) + '.dat')

        wrangler.save_mm(input_path, inputs)
        wrangler.save_mm(output_path, outputs)

        cur_dat_num += 1

## DELETE ##
    print "\nProcessing Test Files.\n"
    test_wav_paths = np.array_split(np.array(test_wav_paths), NUM_DAT_FILES)
    for dat_file in test_wav_paths:
        inputs, outputs = [], []
        for wav_path in dat_file:
            midi_path = wav_path.split('.wav')[0] + '.mid'

        ### DELETE ###
            print "Processing " + wav_path

            sr = _get_sample_rate(config['TRANSFORMS'], args)
            np_input = _transform_track(config, args, wav_path)
            np_output = _generate_expected(config, midi_path, np_input.shape[0], sr)

        ### DELETE ###
            print np_input.shape
            print np_output.shape

            inputs.append(np_input)
            outputs.append(np_output)

        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)

        input_path = os.path.join(paths['test_dir'], str(cur_dat_num) + '.dat')
        output_path = os.path.join(paths['expect_dir'], str(cur_dat_num) + '.dat')

        wrangler.save_mm(input_path, inputs)
        wrangler.save_mm(output_path, outputs)

        cur_dat_num += 1



def _preprocess_config2_subset(config, args, paths, id):
    '''
    Generates processed .wav's and ground truths in experiment directory.
    :param config: dict - config.
    :param args: namespace - passed in during execution.
    :param paths: dict - train, val, test directories for experiment.
    :param id: str - unique id of experiment.
    '''

    # # Preprocess from the subset dir
    # for subdir_name in os.listdir(config['SUBSET_MAPS_DIR']):
    #     subdir_path = os.path.join(config['SUBSET_MAPS_DIR'], subdir_name)
    #
    #     # Check if directory and not file.
    #     if not os.path.isdir(subdir_path):
    #         continue
    #
    #     # Find all .wavs
    #     for dir_parent, _, file_names in os.walk(subdir_path):
    #         for name in file_names:
    #             if name.endswith('.wav'):
    #                 track_name = name.split('.wav')[0]
    #                 midi_name = track_name + '.mid'
    #
    #                 if midi_name in file_names:
    #                     track_path = os.path.join(dir_parent, name)
    #                     midi_path = os.path.join(dir_parent, midi_name)
    #
    #             ### DELETE ###
    #                     print "Processing " + track_path
    #
    #                     # Transform and Generate ground truth
    #                     sr = _get_sample_rate(config['TRANSFORMS'], args)
    #                     np_input = _transform_track(config, args, track_path)
    #                     np_output = _generate_expected(config, midi_path, np_input.shape[0], sr)
    #
    #             ### DELETE ###
    #                     print np_input.shape
    #                     print np_output.shape
    #
    #                     # Save processed .wavs in maps_subset_config2/train to splits/.../train etc.
    #
    #                     ## Key component to Sigtia Configuration 2 ##
    #                     # -> train on synthetic, test on accoustic
    #                     datapoint_id = track_name + '.dat'
    #                     input_path = paths['train_dir']
    #                     test_dirs = config['DATASET_CONFIGS']['config-2_subset']['test']
    #                     if subdir_name in test_dirs:
    #                         input_path = paths['test_dir']
    #
    #                     # Save transform and ground truth
    #                     input_path = os.path.join(input_path, datapoint_id)
    #                     output_path = os.path.join(paths['expect_dir'], datapoint_id)
    #                     wrangler.save_mm(input_path, np_input)
    #                     wrangler.save_mm(output_path, np_output)

    # Get all .wav paths
    train_wav_paths = []
    test_wav_paths = []
    for subdir_name in os.listdir(config['SUBSET_MAPS_DIR']):
        subdir_path = os.path.join(config['SUBSET_MAPS_DIR'], subdir_name)
        if not os.path.isdir(subdir_path):
            continue
        for dir_parent, _, file_names in os.walk(subdir_path):
            for name in file_names:
                if name.endswith('.wav'):
                    track_name = name.split('.wav')[0]
                    midi_name = track_name + '.mid'
                    if midi_name in file_names:
                        wav_path = os.path.join(dir_parent, name)
                        test_dirs = config['DATASET_CONFIGS']['config-2_subset']['test']
                        if subdir_name in test_dirs:
                            test_wav_paths.append(wav_path)
                        else:
                            train_wav_paths.append(wav_path)

    # Shuffle
    np.random.shuffle(train_wav_paths)
    np.random.shuffle(test_wav_paths)

    # Preprocess train .wavs and save into X dat files.

    ## DELETE ##
    print "\nProcessing Training Files.\n"
    cur_dat_num = 0
    train_wav_paths = np.array_split(np.array(train_wav_paths), NUM_DAT_FILES)
    for dat_file in train_wav_paths:
        inputs, outputs = [], []
        for wav_path in dat_file:
            midi_path = wav_path.split('.wav')[0] + '.mid'

            ### DELETE ###
            print "Processing " + wav_path

            sr = _get_sample_rate(config['TRANSFORMS'], args)
            np_input = _transform_track(config, args, wav_path)
            np_output = _generate_expected(config, midi_path, np_input.shape[0], sr)

            ### DELETE ###
            print np_input.shape
            print np_output.shape

            inputs.append(np_input)
            outputs.append(np_output)

        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)

        input_path = os.path.join(paths['train_dir'], str(cur_dat_num) + '.dat')
        output_path = os.path.join(paths['expect_dir'], str(cur_dat_num) + '.dat')

        wrangler.save_mm(input_path, inputs)
        wrangler.save_mm(output_path, outputs)

        cur_dat_num += 1

    ## DELETE ##
    print "\nProcessing Test Files.\n"
    test_wav_paths = np.array_split(np.array(test_wav_paths), NUM_DAT_FILES)
    for dat_file in test_wav_paths:
        inputs, outputs = [], []
        for wav_path in dat_file:
            midi_path = wav_path.split('.wav')[0] + '.mid'

            ### DELETE ###
            print "Processing " + wav_path

            sr = _get_sample_rate(config['TRANSFORMS'], args)
            np_input = _transform_track(config, args, wav_path)
            np_output = _generate_expected(config, midi_path, np_input.shape[0], sr)

            ### DELETE ###
            print np_input.shape
            print np_output.shape

            inputs.append(np_input)
            outputs.append(np_output)

        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)

        input_path = os.path.join(paths['test_dir'], str(cur_dat_num) + '.dat')
        output_path = os.path.join(paths['expect_dir'], str(cur_dat_num) + '.dat')

        wrangler.save_mm(input_path, inputs)
        wrangler.save_mm(output_path, outputs)

        cur_dat_num += 1

def run(config, args, experiment_id):
    '''
    Executes preprocessing based on dataset_config, transform_type, and model specified.
    :param config: dict - config.
    :param args: Namespace - Contains preprocessing specs.
    '''
    print "Preprocessing beginning... for " + experiment_id + '\n'

    # Create directories
    experiment_paths = _create_dest_dirs(experiment_id)

    # Process dataset using specified dataset_config and transform_type
    if args.dataset_config == 'config-2':
        _preprocess_config2(config, args, experiment_paths, experiment_id)
    elif args.dataset_config == 'config-2_subset':
        _preprocess_config2_subset(config, args, experiment_paths, experiment_id)


# def _cqt(config, path):
#     timeSeries, sampleRate = lr.load(path)
#     return lr.cqt(
#         timeSeries,
#         fmin=lr.midi_to_hz(config['FREQ_MIN']),
#         sr=sampleRate,
#         hop_length=config['HOP_LENGTH'],
#         bins_per_octave=config['BINS_PER_8VE'],
#         n_bins=config['N_BINS_TOTAL']
#     )
