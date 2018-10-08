import preprocess as pre
import postprocess as post
from configs import DEFAULT_CONFIG

if __name__ == "__main__":
    print("\nTRANSCRIBE ALL THE MUSIC.\n")
    """
    OLD Interface
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

    """
    NEW Interface
    
    Ex: python app.py -c config2 -t logfilt -m baseline
    
    + Mode
      --> Preprocess
      --> Train/Evaluate
      --> Postprocess
    + Model
      --> Sets type of transform
      --> Sets which model to train/predict
      --> Sets which model to postprocess predictions?
    
    i.e. preprocess + config1_logfilt_baseline 
    = split data based on config1 for baseline model using logfilters
    
    i.e. train/eval + config1_logfilt_baseline
    = checks that data has been preprocessed, then trains/evals baseline model
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

