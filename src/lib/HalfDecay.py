from keras import backend as K
from keras.callbacks import Callback

class HalfDecay(Callback):
    '''
    decay = decay value to subtract each epoch
    '''
    def __init__(self, initial_lr,period):
        super(HalfDecay, self).__init__()
        self.init_lr = initial_lr
        self.period = period

    def on_epoch_begin(self, epoch, logs={}):
        factor = epoch // self.period
        lr  = self.init_lr / (2**factor)
        print("hd: learning rate is now "+str(lr))
        K.set_value(self.model.optimizer.lr, lr)