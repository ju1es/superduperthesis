"""
Notes:
    + For now, validation data lives in the train directories.
      Validation set is created during training from the train directory.
    + Hop_length in ground truth generation is hardcoded.
"""
import os, sys
import madmom as mm
import librosa as lr
import pretty_midi as pm
import numpy as np
from lib import data_wrangler as wrangler

NUM_DAT_FILES = 5

# Hack for PPQ from MAPS
pm.pretty_midi.MAX_TICK = 1e10


def _logfilt(config, track_path):
    """
    Applies logarithmic filterbank stft, normalizing, and windows.
    :param config: dict - config.
    :param track_path: str - path of track to transform.
    :return: np array - transformed track.
    """
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
        fps=config['FPS'],
        circular_shift=config['CIRC_SHIFT'],
        hop_size=config['HOP_SIZE'],
        norm=config['NORM'])

    # Normalize, Rescale
    log_spect = np.array(log_spect) # madmom-class has too many refs to memory.
    log_spect = lr.util.normalize(log_spect, norm=np.inf)

    # Generate Windows:
    min_db = np.min(log_spect)
    log_spect = np.pad(
        log_spect,
        ((config['WINDOW_SIZE'] // 2, config['WINDOW_SIZE'] // 2), (0, 0)),
        'constant', constant_values=min_db)

    windows = []
    for i in range(log_spect.shape[0] - config['WINDOW_SIZE'] + 1):
        w = log_spect[i:i + config['WINDOW_SIZE'], :]
        windows.append(w)

    return np.array(windows)


def _hcqt(config, track_path):
    """
    Applies HCQT, normalizes, and windows.

    HCQT Authors: Bittner and McPhee

    :param config: dict - config.
    :param track_path: str - path of track to transform.
    :return: np array - transformed track.
    """
    # Load
    y, sr = lr.load(track_path, sr=config['SR'])

    # HCQT
    cqt_list = []
    shapes = []
    harmonics = config['HARMONICS']
    for h in harmonics:
        cqt = lr.cqt(
            y,
            sr=sr,
            hop_length=config['HOP_LENGTH'],
            fmin=config['FMIN'] * float(h),
            n_bins=config['BINS_PER_OCTAVE'] * config['N_OCTAVES'],
            bins_per_octave=config['BINS_PER_OCTAVE'])
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    log_hcqt = ((1.0 / 80.0) * lr.core.amplitude_to_db(
        np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    # Normalize
    log_hcqt = log_hcqt.T # Time, Freq, Harmonics
    log_hcqt = lr.util.normalize(log_hcqt, norm=np.inf)

    # Windows
    min_db = np.min(log_hcqt)
    log_hcqt = np.pad(
        log_hcqt,
        ((config['WINDOW_SIZE']//2, config['WINDOW_SIZE']//2), (0, 0), (0, 0)),
        'constant',
        constant_values=min_db)

    windows = []
    for i in range(log_hcqt.shape[0] - config['WINDOW_SIZE'] + 1):
        w = log_hcqt[i:i+config['WINDOW_SIZE'], :]
        windows.append(w)

    return np.array(windows)


def _generate_expected(config, midi_path, input_shape, sr, hop_length):
    """
    Generates expected array off of associative midi.
    :param config: config.
    :param midi_path: path to midi generate expected off of.
    :param input_shape: shape of input.
    :param sr: sample rate.
    :param hop_length: hop length.
    :return: np array - ground truth.
    """
    pm_midi = pm.PrettyMIDI(midi_path)
    times = lr.frames_to_time(
                np.arange(input_shape),
                sr=sr,
                hop_length=hop_length)
    expected = pm_midi.get_piano_roll(
                        fs=sr,
                        times=times)[config['MIN_MIDI']:config['MAX_MIDI']+1].T
    expected[expected > 0] = 1
    return expected


def _transform_track(config, args, track_path):
    X = []
    if args.transform_type == 'logfilt':
        X = _logfilt(config['TRANSFORMS']['logfilt'], track_path)
    elif args.transform_type == 'hcqt':
        X = _hcqt(config['TRANSFORMS']['hcqt'], track_path)

    return X


def _get_sr_and_hl(transform_config, args):
    sr = 0
    hl = 0
    if args.transform_type == 'logfilt':
        sr, hl = transform_config['logfilt']['SR'], transform_config['logfilt']['HOP_SIZE']
    elif args.transform_type == 'hcqt':
        sr, hl = transform_config['hcqt']['SR'], transform_config['hcqt']['HOP_LENGTH']

    return sr, hl


def _transform_wavs(cur_dat_num, dir_type, wav_paths, config, args, paths):
    """
    Transform wavs in a specified directory
    :param dir_type: str - train/test
    :param wav_paths: str - wav paths.
    :param config: dict - specs.
    :param args: namespace - specs of run.
    :param paths: run specific split paths.
    """
    dat_num = cur_dat_num
    for dat_file in wav_paths:
        inputs, outputs = [], []
        for wav_path in dat_file:
            midi_path = wav_path.split('.wav')[0] + '.mid'

            print "Processing " + wav_path

            sr, hl = _get_sr_and_hl(config['TRANSFORMS'], args)
            np_input = _transform_track(config, args, wav_path)
            np_output = _generate_expected(config, midi_path, np_input.shape[0], sr, hl)

            ### Sanity Check ###
            print np_input.shape
            print np_output.shape

            inputs.append(np_input)
            outputs.append(np_output)

        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)

        input_path = os.path.join(paths[dir_type], str(dat_num) + '.dat')
        output_path = os.path.join(paths['expect'], str(dat_num) + '.dat')

        wrangler.save_mm(input_path, inputs)
        wrangler.save_mm(output_path, outputs)

        dat_num += 1

    return dat_num


def _preprocess_config2(config, args, paths, id):
    """
    Generates processed .wav's and ground truths in experiment directory.
    :param config: dict - config.
    :param args: namespace - passed in during execution.
    :param paths: dict - train, val, test directories for experiment.
    :param id: str - unique id of experiment.
    """

    # Fetch .wav paths
    train_wav_paths, test_wav_paths = wrangler.fetch_config2_paths(config)

    # Shuffle
    np.random.shuffle(train_wav_paths)
    np.random.shuffle(test_wav_paths)

    print "\nProcessing Training Files.\n"

    # Split into N files (i/o reasons)
    train_wav_paths = np.array_split(np.array(train_wav_paths), NUM_DAT_FILES)
    test_wav_paths = np.array_split(np.array(test_wav_paths), NUM_DAT_FILES)

    # Transform wavs and save
    cur_dat_num = 0
    cur_dat_num = _transform_wavs(cur_dat_num, 'train', train_wav_paths, config, args, paths)
    _transform_wavs(cur_dat_num, 'test', test_wav_paths, config, args, paths)


def run(config, args, dataset_id):
    """
    Executes preprocessing based on dataset_config, transform_type, and model specified.
    :param config: dict - config.
    :param args: Namespace - Contains preprocessing specs.
    :param experiment_id: str - id.
    """
    print "Preprocessing beginning... for " + dataset_id + '\n'

    # Create directories
    dataset_paths = wrangler.create_split_dirs(dataset_id)

    # Process dataset using specified dataset_config and transform_type
    if args.dataset_config == 'config-2':
        _preprocess_config2(config, args, dataset_paths, dataset_id)
    else:
        print 'ERROR: ' + args.dataset_config + ' doesn\'t exist.'
