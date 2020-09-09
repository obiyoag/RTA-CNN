import os
import numpy as np
# set tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import cli
from utils import *
import architectures

from keras.callbacks import Callback
from keras import optimizers
from keras.layers.merge import concatenate
from keras.callbacks import ReduceLROnPlateau
from keras import  backend as K
from keras.models import load_model
import tensorflow as tf

args = None
tf.logging.set_verbosity(tf.logging.ERROR)
logger = None


def train():

    train_genrator = Generaor(train_fold, args.batch_size)
    test_genrator = Generaor(test_fold, args.batch_size)

    adam = optimizers.adam(lr=args.lr)
    model_factory = architectures.__dict__[args.arch]
    model = model_factory()
    model.compile(optimizer = 'adam', loss=architectures.en_loss, metrics = ["accuracy"])

    if args.summary:
        model.summary()

    class MyCbk(Callback):
        def __init__(self, model):
            self.model_to_save = model
        def on_epoch_end(self, epoch, logs=None):
            if epoch >= args.epoch2save:
                print('save model_at_epoch_%d.h5' % epoch)
                self.model_to_save.save(ex_path + '/models/model_%d.h5' % epoch)
    
    cbk = MyCbk(model)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=5, verbose=0, mode='auto', cooldown=0, min_lr=1e-7)
    results = model.fit_generator(train_genrator.get_data(), 
              steps_per_epoch = train_genrator.data_num/args.batch_size,
              validation_data = test_genrator.get_data(),
              validation_steps = test_genrator.data_num/args.batch_size,
              epochs = args.epochs, verbose=1, callbacks=[cbk, reduceLR])

    return results

def plot_and_save(results):
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    epochs = range(1, len(loss)+1)

    # plot and save the loss curve
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(ex_path + "/loss.png", dpi=1000)
    print("loss curve has been saved")

    # save the loss values
    with open(ex_path + "/logs.txt", 'a+') as f:
        f.write("Training loss: " + str(loss) + "\n")
        f.write("Validation loss: " + str(val_loss) + "\n")
    print("loss values has been saved")

def test():

    af_generator = Generaor(test_fold, args.batch_size, 'AF')
    normal_generator = Generaor(test_fold, args.batch_size, 'normal')
    other_generator = Generaor(test_fold, args.batch_size, 'other')
    
    af_list = []
    normal_list = []
    other_list = []
    total_list = []

    for id in range(args.epoch2save, args.epochs):

        model =  load_model(ex_path + '/models/model_'+str(id)+'.h5', custom_objects={'en_loss': architectures.en_loss})

        af_score = model.evaluate_generator(generator=af_generator.get_data(), steps=af_generator.data_num//args.batch_size)
        normal_score = model.evaluate_generator(generator=normal_generator.get_data(), steps=normal_generator.data_num//args.batch_size)
        other_score = model.evaluate_generator(generator=other_generator.get_data(), steps=other_generator.data_num//args.batch_size)
        total_score = (af_score[1]+normal_score[1]+other_score[1])/3

        af_list.append(af_score[1])
        normal_list.append(normal_score[1])
        other_list.append(other_score[1])
        total_list.append(total_score)

        logger.info(ex_path + '/models/model_' + str(id) + "\t AF:{:.5f}   normal:{:.5f}   other:{:.5f}   total:{:.5f}"
        .format(af_score[1], normal_score[1], other_score[1], total_score))

        K.clear_session()
        tf.reset_default_graph()
    
    best_index = total_list.index(max(total_list))
    logger.info("The best model for ex" + str(args.experiment_index) + "\t AF:{:.5f}   normal:{:.5f}   other:{:.5f}   total:{:.5f}"
    .format(af_list[best_index], normal_list[best_index], other_list[best_index], total_list[best_index]))


if __name__ == "__main__":

    K.clear_session()
    tf.reset_default_graph()

    args = cli.parse_commandline_args()
    ex = args.experiment_index
    assert ex is not None

    train_fold, test_fold = get_folds(ex)
    ex_path = "logs/ex" + str(ex)
    log_path = ex_path + "/logs.txt"

    # delete the logs of the last training
    if os.path.exists(log_path):
        os.remove(log_path)

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    session = tf.Session(config=config)
    K.set_session(session )
    
    logger = get_logger(log_path)

    results = train()
    plot_and_save(results)
    test()
