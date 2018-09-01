DEFAULT_CONFIG = {
    'MODE' : {
        'PREPROCESS' : {
            'TRAINING_SET_PATH' : 'training_sets/',
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