import sys
import tensorflow as tf
import train
from configs import CONFIG
from datetime import datetime

DEVICE_NAME = "/gpu:0"

class Namespace:
    def __init__(selfself, **kwargs):
        self.__dict__.update(kwargs)

# with tf.device(DEVICE_NAME):
#     start_time = datetime.now()
#
#     # Call training file here
#     args =  Namespace(mode='train',
#                       dataset_config='config-2',
#                       transform_type='logfilt',
#                       model='baseline')
#     experiment_id = 'config-2_logfilt_baseline'
#     train.run(CONFIG, args,experiment_id)
#
#     print "\n\n\n\n\n"
#     print "Time taken: " + str(datetime.now() - start_time)
#     print "\n\n\n\n\n"
#

device_name = "gpu"  # Choose device from cmd line. Options: gpu or cpu
shape = (4, 4)
if device_name == "gpu":
     device_name = "/gpu:0"
else:
     device_name = "/cpu:0"

with tf.device(device_name):
     random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
     dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
     sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
         result = session.run(sum_operation)
         print(result)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)