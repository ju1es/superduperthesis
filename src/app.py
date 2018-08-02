import preprocess as pre
import postprocess as post

if __name__ == "__main__":
    print("Transcribe das music.")
    """
    + Process each signal (song)
    + Train model(s)
    + Evaluate model(s)
    + Visualize --> wav2mid
    """

    pre.test_cqt()

    """
    + Handle args
    + --> Process datapoints and output into training samples
    """
