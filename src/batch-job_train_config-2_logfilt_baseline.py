import sys
import tensorflow as tf
import train
from configs import CONFIG
from datetime import datetime

DEVICE_NAME = "/gpu:0"

class Namespace:
    def __init__(selfself, **kwargs):
        self.__dict__.update(kwargs)

with tf.device(DEVICE_NAME):
    start_time = datetime.now()

    # Call training file here
    args =  Namespace(mode='train',
                      dataset_config='config-2',
                      transform_type='logfilt',
                      model='baseline')
    experiment_id = 'config-2_logfilt_baseline'
    train.run(CONFIG, args,experiment_id)

    print "\n\n\n\n\n"
    print "Time taken: " + str(datetime.now() - start_time)
    print "\n\n\n\n\n"