import os
import numpy as np
import matplotlib.pyplot as plt 
from keras.callbacks import Callback
from keras import optimizers
from keras.losses import categorical_crossentropy
from keras.layers.merge import concatenate
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from keras import  backend as K
import tensorflow as tf
import cli
from utils import *
from architectures import RTA_CNN


# gpu setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.Session(config=config)
K.set_session(session )


args = None
logger = None


def train():

    train_genrator = Generaor(train_folder, args.batch_size)
    test_genrator = Generaor(test_folder, args.batch_size)

    adam = optimizers.adam(lr=args.lr)
    model_factory = architectures.__dict__[args.arch]
    model = model_factory()
    model.compile(optimizer = 'adam', loss=focal_loss, metrics = ["accuracy"])
    model.summary()

    class MyCbk(Callback):
        def __init__(self, model):
            self.model_to_save = model
        def on_epoch_end(self, epoch, logs=None):
            if epoch>=80:
                print('save model_at_epoch_%d.h5' % epoch)
                self.model_to_save.save(model_path + '/model_%d.h5' % epoch)
    
    cbk = MyCbk(model)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=5, verbose=0, mode='auto', cooldown=0, min_lr=1e-7)
    results = model.fit_generator(train_genrator.get_data(), 
              steps_per_epoch = train_genrator.data_num/args.batch_size,
              validation_data = test_genrator.get_data(),
              validation_steps = test_genrator.data_num/args.batch_size,
              epochs = args.epoch, verbose=1, callbacks=[cbk, reduceLR])

    plot(results)

def test():
    af_generator = Generaor(test_folder, args.batch_size, 'AF')
    normal_generator = Generaor(test_folder, args.batch_size, 'normal')
    other_generator = Generaor(test_folder, args.batch_size, 'other')

    for id in range(80,100):

        model =  load_model(model_path + '/model_'+str(id)+'.h5', custom_objects={'focal_loss': focal_loss})

        af_score = model.evaluate_generator(generator=af_generator.get_data(), steps=af_generator.data_num//args.batch_size)
        normal_score = model.evaluate_generator(generator=normal_generator.get_data(), steps=normal_generator.data_num//args.batch_size)
        other_score = model.evaluate_generator(generator=other_generator.get_data(), steps=other_generator.data_num//args.batch_size)

        logger.info(model_path + '/model_' + str(id) + "\t AF:%.5f   normal:%.5f   other:%.5f   total:%.5f"
        .format(af_score[1], normal_score[1], other_score[1], (af_score[1]+normal_score[1]+other_score[1])/3))

        K.clear_session()
        tf.reset_default_graph()


if __name__ == "__main__":
    args = cli.parse_commandline_args()
    ex = args.experiment_index
    assert ex is not None

    train_folder, test_folder = get_folders(ex)
    model_path = "models/ex" + str(ex)
    log_path = "logs/ex" + str(ex) + ".txt"
    logger = get_logger(log_path)

    train()
    test()