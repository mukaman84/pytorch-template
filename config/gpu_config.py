import tensorflow as tf
from tensorflow.keras import backend as K
try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    print(ModuleNotFoundError)

def gpu_config():
    try:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))
    except:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        # config.gpu_options.allow_soft_placement = True
        session = tf.Session(config=config)
        K.set_session(session)