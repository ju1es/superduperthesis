DEFAULT_CONFIG = {
    'MODE' : {
        'PREPROCESS' : {
            'TRAINING_SET_PATH' : 'splits/',
            'CONFIG-2_LOGFILT_BASELINE' : {
                'TRAIN_DIR' : 'config-2_logfilt_baseline/train/',
                'TEST_DIR' : 'config-2_logfilt_baseline/test/',
                'NUM_BANDS' : 48,
                'NUM_CHANNELS' : 1,
                'SR' : 44100,
                'UNIQUE_FILTERS' : True,
                'NORM_FILTERS' : True,
                'FRAME_SIZE' : 4096,
                'FFT_SIZE' : 4096,
                'FMIN' : 30,
                'FMAX' : 8000.0,
                'FREF' : 440.0,
                'CIRC_SHIFT' : False,
                'FPS' : 31.25
            },
            'CQT' : {
                'FREQ_MIN' : 21,
                'SAMPLE_RATE' : 22050,
                'HOP_LENGTH' : 512,
                'BINS_PER_8VE' : 24,
                'N_BINS_TOTAL' : 24 * 8
            },
            'STFTTRI' : {
                'NUM_BANDS' : 24,
                'NUM_CHANNELS' : 1,
                'UNIQUE_FILTERS' : False,
                'SAMPLE_RATE' : 44100

            }
        }
    },
}