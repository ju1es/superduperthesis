import preprocess as pre
import postprocess as post
from configs import DEFAULT_CONFIG

if __name__ == "__main__":
    print("\nTRANSCRIBE ALL THE MUSIC.\n")
    """
    + Ingest config-type from command line
        --> MODE
            | PREPROCESS
                + transform-type
                    --> CQT
                        + freq_min
                        + sample_rate
                        + hop_length
                        + bins_per_8ve
                        + n_bins
                    --> STFT + Tri
            | TRAIN
                + model
            | EVAL
                + model file
            | POSTPROCESS
                + ...
    + PREPROCESS
        --> Output np arrays of transformed signals
    """

    #
    # Hardcoded
    #

    MODE = 'PREPROCESS'
    TRANSFORM_TYPE = 'STFTTRI'
    # DATASET_PATH = "datasets/ISOL/CH/MAPS_ISOL_CH0.1_F_SptkBGAm.wav"
    DATASET_PATH = "datasets/ISOL/CH"

    if MODE == 'PREPROCESS':
        pre.run(
            DEFAULT_CONFIG['MODE']['PREPROCESS'],
            TRANSFORM_TYPE,
            DATASET_PATH)

