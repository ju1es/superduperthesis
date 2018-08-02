# import madmom as mm
import librosa as lr
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt

TEST_DATAPOINT = './datasets/Testing/MAPS_RAND_P2_M21-108_I32-96_S0_n12_SptkBGAm.wav'

def test_cqt():
    # Load datapoint
    timeSeries, sampleRate = lr.load(TEST_DATAPOINT)
    cqtSpect = lr.cqt(timeSeries,
                      fmin=lr.midi_to_hz(21),
                      sr=sampleRate,
                      hop_length=512,
                      bins_per_octave=24,
                      n_bins=24 * 8)
    lr.display.specshow(lr.amplitude_to_db(cqtSpect, ref=np.max),
                        sr=sampleRate,
                        x_axis='time',
                        y_axis='cqt_note',
                        fmin=lr.midi_to_hz(21),
                        bins_per_octave=24)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    # plt.tight_layout()
    plt.show()