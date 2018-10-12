'''
sampleMaps - Gets random ~250mb of MAPs data
--> Config 2: Train (...), Test (ENSTDkAm, ENSTDkCl)

'''
import os
import numpy as np
import shutil

DATASET_DIR = './datasets/maps'
MAPS_SUBSET_CONFIG_2_DIR = './datasets/maps_subset_config2/'
MAPS_SUBSET_CONFIG_2_DIR_TRAIN = os.path.join(MAPS_SUBSET_CONFIG_2_DIR, 'train')
MAPS_SUBSET_CONFIG_2_DIR_TEST = os.path.join(MAPS_SUBSET_CONFIG_2_DIR, 'test')
TEST_DIRS = ['ENSTDkAm', 'ENSTDkCl']

if __name__ == "__main__":

    # Setup directories
    if not os.path.exists(MAPS_SUBSET_CONFIG_2_DIR):
        os.mkdir(MAPS_SUBSET_CONFIG_2_DIR)
    if not os.path.exists(MAPS_SUBSET_CONFIG_2_DIR_TRAIN):
        os.mkdir(MAPS_SUBSET_CONFIG_2_DIR_TRAIN)
    if not os.path.exists(MAPS_SUBSET_CONFIG_2_DIR_TEST):
        os.mkdir(MAPS_SUBSET_CONFIG_2_DIR_TEST)

    '''
    Traverse (...) for Train Separately
    --> Calculate total MB size AND total # of .wavs
    --> Calculate 1/10 of randomly sampled data
    
    --
    
    Randomly sample X% and copy .wav, .txt, and .midi to sampleMaps/train
    '''
    total_size = 0
    track_wav_paths = []
    track_midi_paths = []
    track_txt_paths = []

    for subdir_name in os.listdir(DATASET_DIR):
        subdir_path = os.path.join(DATASET_DIR, subdir_name)

        if not os.path.isdir(subdir_path) or subdir_name in TEST_DIRS:
            continue

        for dir_parent, dir_name, file_names in os.walk(subdir_path):
            for name in file_names:
                if name.endswith('.wav'):
                    track_name = name.split('.wav')[0]
                    midi_name = track_name + '.mid'
                    txt_name = track_name + '.txt'

                    if midi_name in file_names:
                        track_path = os.path.join(dir_parent, name)
                        total_size += os.path.getsize(track_path)
                        track_wav_paths.append(track_path)
                        track_midi_paths.append(os.path.join(dir_parent, midi_name))
                        track_txt_paths.append(os.path.join(dir_parent, txt_name))

    print "Total train tracks: " + str(len(track_wav_paths))
    total_size = total_size / 1000000.0
    print "Total train size (MB): " + str(total_size)

    # Shuffle track_paths and get % and re-calculate total num and size of tracks
    np.random.shuffle(track_wav_paths)
    split_index = int(len(track_wav_paths) * 0.1)
    track_wav_paths = track_wav_paths[:split_index]
    total_size = 0
    for path in track_wav_paths:
        total_size += os.path.getsize(path)

    print "Total sampled train tracks: " + str(len(track_wav_paths))
    total_size = total_size / 1000000.0
    print "Total sampled train size (MB): " + str(total_size)

    files_to_copy = track_wav_paths + track_midi_paths + track_txt_paths
    for path in files_to_copy:
        # Copy samples to dir
        shutil.copy(path, MAPS_SUBSET_CONFIG_2_DIR_TRAIN)


    print "Copied to " + MAPS_SUBSET_CONFIG_2_DIR_TRAIN

    '''
    Traverse (ENSTDkAm, ENSTDkCl) for Test Separately
    --> Calculate total MB size
    --> Calculate 1/10 of randomly sampled data

    --

    Randomly sample X% and copy .wav, .txt, and .midi to sampleMaps/test
    '''
    # num_tracks = 0
    # total_size = 0
    # for subdir_name in os.listdir(DATASET_DIR):
    #     subdir_path = os.path.join(DATASET_DIR, subdir_name)
    #
    #     if not os.path.isdir(subdir_path) or subdir_name in TEST_DIRS:
    #         continue
    #
    #     for dir_parent, dir_name, file_names in os.walk(subdir_path):
    #         for name in file_names:
    #             num_tracks += 1
    #             track_path = os.path.join(dir_parent, name)
    #             total_size += os.path.getsize(track_path)
    #
    # print "Total test tracks: " + str(num_tracks)
    # total_size = total_size / 1000000.0
    # print "Total test size (MB): " + str(total_size)
