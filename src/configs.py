CONFIG = {
    'MIN_MIDI' : 21,
    'MAX_MIDI' : 108,
    'DATASET_DIR' : 'datasets/maps/',
    'MAPS_CONFIG2_FOLD_1' : 'datasets/maps_config2/fold_1/',
    'MAPS_SUBSET_CONFIG2' : 'datasets/maps_subset_config2/',
    'MAPS_CONFIG1_FOLD_3' : 'datasets/maps_config1/fold_3/',
    'DATASET_CONFIGS' : {
        'config-2' : {
                'test' : ['ENSTDkAm', 'ENSTDkCl']
        },
        'maps_subset_config2' : {
                'test' : ['test']
        },
        'maps_config1_fold_3' : {
                'test' : ['test']
        },
        'maps_config2_fold_1_adsr' : {
                'test' : ['test']
        }
    },
    'TRANSFORMS' : {
        'logfilt_shallow' : {
                'NUM_BANDS' : 36,
                'NUM_CHANNELS' : 1,
                'SR' : 22050,
                'UNIQUE_FILTERS' : True,
                'NORM_FILTERS' : True,
                'FRAME_SIZE' : 4096,
                'FFT_SIZE' : 4096,
                'FMIN' : 27.5,
                'FMAX' : 8000.0,
                'FREF' : 440.0,
                'FPS' : 100,
                'CIRC_SHIFT' : False,
                'HOP_SIZE' : 256.0,
                'NORM' : True
        },
        'logfilt' : {
                'WINDOW_SIZE' : 5,
                'NUM_BANDS' : 48,
                'NUM_CHANNELS' : 1,
                'SR' : 44100,
                'UNIQUE_FILTERS' : True,
                'NORM_FILTERS' : True,
                'FRAME_SIZE' : 4096,
                'FFT_SIZE' : 4096,
                'FMIN' : 27.5,
                'FMAX' : 8000.0,
                'FREF' : 440.0,
                'FPS' : 100,
                'CIRC_SHIFT' : False,
                'HOP_SIZE' : 441.0,
                'NORM' : True
        },
        'logfilt_adsr_baseline' : {
                'WINDOW_SIZE' : 11,
                'NUM_BANDS' : 48,
                'NUM_CHANNELS' : 1,
                'SR' : 44100,
                'UNIQUE_FILTERS' : True,
                'NORM_FILTERS' : True,
                'FRAME_SIZE' : 4096,
                'FFT_SIZE' : 4096,
                'FMIN' : 27.5,
                'FMAX' : 2325.0,
                'FREF' : 440.0,
                'FPS' : 100,
                'CIRC_SHIFT' : False,
                'HOP_SIZE' : 441.0,
                'NORM' : True
        },
        'hcqt_shallow' : {
                'BINS_PER_OCTAVE' : 12,
                'N_OCTAVES' : 6,
                'HARMONICS' : [0.5, 1, 2, 3, 4, 5],
                'SR' : 44100,
                'FMIN' : 27.5,
                'HOP_LENGTH' : 256
        },
        'hcqt' : {
                'WINDOW_SIZE' : 5,
                'BINS_PER_OCTAVE' : 60,
                'N_OCTAVES' : 6,
                'HARMONICS' : [0.5, 1, 2, 3, 4, 5],
                'SR' : 22050,
                'FMIN' : 27.5,
                'HOP_LENGTH' : 256
        },
        'cqt_shallow' : {
                'BINS_PER_OCTAVE' : 12,
                'N_OCTAVES' : 8,
                'SR' : 22050,
                'FMIN' : 27.5,
                'HOP_LENGTH' : 512
        }
    },
    'MODELS' : {
        'baseline' : {
            'TRAIN' : {
                'EPOCHS' : 50,
                'BATCH_SIZE' : 256,
                'LR' : 0.1,
                'HALVING_N_EPOCHS' : 5,
                'MOMENTUM' : 0.9
            }
        },
        'baseline-checkpoint' : True, # For Evaluation
        'shallow_net' : {
            'TRAIN' : {
                'EPOCHS' : 5,
                'BATCH_SIZE' : 256,
                'LR' : 0.1,
                'HALVING_N_EPOCHS' : 10,
                'MOMENTUM' : 0.9
            }
        },
        'shallow_net_checkpoint' : True, # For Evaluation
        'hcqt_shallow_net' : {
            'TRAIN' : {
                'EPOCHS' : 5,
                'BATCH_SIZE' : 256,
                'LR' : 0.1,
                'HALVING_N_EPOCHS' : 10,
                'MOMENTUM' : 0.9
            }
        },
        'hcqt_shallow_net_checkpoint' : True, # For Evaluation
        'hcqt-conv' : {
            'TRAIN' : {
                'EPOCHS' : 50,
                'BATCH_SIZE' : 256,
                'LR' : 0.1,
                'HALVING_N_EPOCHS' : 5,
                'MOMENTUM' : 0.9
            }
        },
        'hcqt-conv-checkpoint' : True,
        'adsr_net_baseline' : {
            'TRAIN' : {
                'EPOCHS' : 50,
                'BATCH_SIZE' : 256,
                'LR' : 0.1,
                'HALVING_N_EPOCHS' : 5,
                'MOMENTUM' : 0.9
            }
        },
        'adsr_net_baseline-checkpoint' : True # For Evaluation
    }
    # 'MODE' : {
    #     'PREPROCESS' : {
    #         'TRAINING_SET_PATH' : 'splits/',
    #         'CONFIG-2_LOGFILT_BASELINE' : {
    #             'TRAIN_DIR' : 'config-2_logfilt_baseline/train/',
    #             'TEST_DIR' : 'config-2_logfilt_baseline/test/',
    #             'NUM_BANDS' : 48,
    #             'NUM_CHANNELS' : 1,
    #             'SR' : 44100,
    #             'UNIQUE_FILTERS' : True,
    #             'NORM_FILTERS' : True,
    #             'FRAME_SIZE' : 4096,
    #             'FFT_SIZE' : 4096,
    #             'FMIN' : 30,
    #             'FMAX' : 8000.0,
    #             'FREF' : 440.0,
    #             'CIRC_SHIFT' : False,
    #             'FPS' : 31.25
    #         },
    #         'CQT' : {
    #             'FREQ_MIN' : 21,
    #             'SAMPLE_RATE' : 22050,
    #             'HOP_LENGTH' : 512,
    #             'BINS_PER_8VE' : 24,
    #             'N_BINS_TOTAL' : 24 * 8
    #         },
    #         'STFTTRI' : {
    #             'NUM_BANDS' : 24,
    #             'NUM_CHANNELS' : 1,
    #             'UNIQUE_FILTERS' : False,
    #             'SAMPLE_RATE' : 44100
    #
    #         }
    #     }
    # },
}