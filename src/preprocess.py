# import madmom as mm
import librosa as lr
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt
from lib import dataWrangler as wrangler


def _cqt(config, filePath):
    timeSeries, sampleRate = lr.load(filePath)
    return lr.cqt(
        timeSeries,
        fmin=lr.midi_to_hz(config['FREQ_MIN']),
        sr=sampleRate,
        hop_length=config['HOP_LENGTH'],
        bins_per_octave=config['BINS_PER_8VE'],
        n_bins=config['N_BINS_TOTAL']
    )


def _apply_cqt(config, datasetPath):
    '''
    Fetches datapoints. Applies CQT. Outputs to training dir.

    Params
    ------
    config : dict
        Sets CQT spec.

    dataset_path : str
        Root of dataset to traverse.

    Returns
    -------
    void
    '''
    datapointsPaths = wrangler.fetchPaths(datasetPath)
    for path in datapointsPaths:
        cqt_spect = _cqt(config['CQT'], path)
        wrangler.saveToTraining(config, cqt_spect, path)


def run(config, type, datasetPath):
    if type == 'CQT':
        print("\nRunning CQT...\n")
        _apply_cqt(config, datasetPath)
