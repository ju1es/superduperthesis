import os
import numpy as np

def fetchPaths(datasetPath):
    paths = []
    for root, dirs, files in os.walk(datasetPath):
        for file in files:
            if file.endswith(".wav"):
                paths.append(os.path.join(root, file))
    return paths

def saveToTraining(config, datapoint, datapointPath):

    # TODO:
    # + For each column in transform
    # --> Reshape.
    #     + np.abs, rescale to (0 to 1)
    # --> Generate ground truth array
    #     + np.arange(Nframes) * t
    #     ---> Where t = hopLength/sampleRate
    # --> Use associative .txt to check which pitches are on or off.

    _, filename = os.path.split(datapointPath)
    destination = config['TRAINING_SET_PATH'] + os.path.splitext(filename)[0] + ".npy"
    print("Saved " + destination + "...")
    np.save(destination, datapoint)