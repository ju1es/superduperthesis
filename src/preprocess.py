import madmom as mm
import librosa as lr
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt
from lib import dataWrangler as wrangler


def run(config, type, datasetPath):
    '''
    Generates ground truths.

    Params
    ------
    config : dict
        Config.

    type : str
        Type of transform to apply to .wav's

    datasetPath : str
        Root directory of dataset


    '''
    datapointsPaths = wrangler.fetchPaths(datasetPath)
    print(len(datapointsPaths))

    # if type == 'CQT':
    #     print("\nRunning CQT...\n")
    #     _apply_cqt(config, datapointsPaths)
    # elif type == 'STFTTRI':
    #     print("\nRunning STFT + Tri Filter...\n")
    #     _apply_stfttri(config, datapointsPaths)


def _apply_cqt(config, paths):
    '''
    Applies CQT to each datapoint. Outputs to training dir.

    Params
    ------
    config : dict
        Sets CQT spec.

    paths : str
        Paths of all datapoints.

    Returns
    -------
    void
    '''
    for path in paths:
        cqt_spect = _cqt(config['CQT'], path)
        wrangler.saveToTraining(config, cqt_spect, path)

def _apply_stfttri(config, paths):
    '''
    Applies stft + tri filter to each datapoint. Outputs to training dir.

    Params
    ------
    config : dict
        Sets CQT spec.

    paths : str
        Paths of all datapoints.

    Returns
    -------
    void
    '''
    for path in paths:
        stfttri_spect = _stfttri(config['STFTTRI'], path)
        wrangler.saveToTraining(config, stfttri_spect, path)


def _cqt(config, path):
    timeSeries, sampleRate = lr.load(path)
    return lr.cqt(
        timeSeries,
        fmin=lr.midi_to_hz(config['FREQ_MIN']),
        sr=sampleRate,
        hop_length=config['HOP_LENGTH'],
        bins_per_octave=config['BINS_PER_8VE'],
        n_bins=config['N_BINS_TOTAL']
    )


def _stfttri(config, path):
    return mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
        path,
        num_bands=config['NUM_BANDS'],
        num_channels=config['NUM_CHANNELS'],
        unique_filters=config['UNIQUE_FILTERS']).T