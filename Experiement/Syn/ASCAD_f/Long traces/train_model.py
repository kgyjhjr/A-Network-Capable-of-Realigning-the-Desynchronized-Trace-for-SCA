import keras
import h5py
import os
import tensorflow as tf

import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.utils import *
# put root project folder path here:
sys.path.append('E:\jingtong\Code')

os.environ["OMP_NUM_THREADS"] = '2'  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '2'  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '2'  # export MKL_NUM_THREADS=6


import importlib

from Dataset.Syn.ReadASCADf_long import ReadASCADf
from Dataset.dataset_parameters import *
from Sca_metrics.sca_metrics import sca_metrics
from Wavelet_scattering_transform.Scattering.keras import Scattering1D
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
def WSMCTCNN_model(classes, number_of_samples):
    batch_size = 800
    inputs = Input(shape= number_of_samples)
    x =Scattering1D(J= 3, max_order=2 )(inputs)

    x = Conv1D(kernel_size=10, strides=5, filters=16, activation='selu', padding='same')(x)
    x = MaxPool1D(pool_size=2, strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(kernel_size=16, strides=8, filters=32, activation='selu', padding='same')(x)
    x = MaxPool1D(pool_size=4, strides=4, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(kernel_size=12, strides=6, filters=64, activation='selu', padding='same')(x)
    x = MaxPool1D(pool_size=2, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(400, activation='selu', kernel_initializer='random_uniform')(x)
    x = Dense(400, activation='selu', kernel_initializer='random_uniform')(x)
    x = Dense(400, activation='selu', kernel_initializer='random_uniform')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='cnn_scatter')
    model.summary()
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model,batch_size


if __name__ == "__main__":
    leakage_model = "ID"
    model_name = "cnn"
    feature_selection_type = "NOPOI"
    npoi = 30000
    target_byte = 2


    """ Parameters for the analysis """
    classes = 256

    ascadf_parameters = ascadf
    n_profiling = ascadf_parameters["n_profiling"]        #50000
    n_attack = ascadf_parameters["n_attack"]                #5000
    n_validation = ascadf_parameters["n_validation"]        #5000
    n_attack_ge = ascadf_parameters["n_attack_ge"]          #3000
    n_validation_ge = ascadf_parameters["n_validation_ge"]  #3000

    """ Create dataset for ASCADf """
    ascad_dataset = ReadASCADf(
        n_profiling,
        n_attack,
        n_validation,
        file_path=f"E:\\jingtong\\ASCAD\\ASCAD\\ASCADf\\ATMega8515_raw_traces_200000.h5",
        target_byte=target_byte,
        leakage_model=leakage_model,
        first_sample=0,
        number_of_samples=npoi,
        reshape_to_cnn= True,
    )

    """ Open best model """
    #Loading untrained model
    model, batch_size = WSMCTCNN_model(classes, npoi)
    # Loading trained model weights
    model.load_weights('E:\jingtong\Code\Pre-trained model\ASCADf_long_traces\desync0\my_model.h5')
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    """ Train model """
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        period=1)
    model.save_weights(checkpoint_path.format(epoch=0))
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.001)
    # history = model.fit(
    #     x=ascad_dataset.x_profiling,
    #     y=ascad_dataset.y_profiling,
    #     batch_size=batch_size,
    #     verbose=2,
    #     epochs=90,
    #     shuffle=True,
    #     validation_data=(ascad_dataset.x_validation, ascad_dataset.y_validation),
    #     callbacks=[reduce_lr])

    model.save('my_model.h5')
    """ Get DL metrics """
    # accuracy = history.history["accuracy"]
    # val_accuracy = history.history["val_accuracy"]
    # loss = history.history["loss"]
    # val_loss = history.history["val_loss"]

    """ Compute GE, SR and NT for attack set """
    ge_attack, sr_attack, nt_attack = sca_metrics(
        model, ascad_dataset.x_attack, n_attack_ge, ascad_dataset.labels_key_hypothesis_attack, ascad_dataset.correct_key)
    print(f"GE attack: {ge_attack[n_attack_ge - 1]}")
    print(f"SR attack: {sr_attack[n_attack_ge - 1]}")
    print(f"Number of traces to reach GE = 1: {nt_attack}")
    x = range(len(ge_attack))
    plt.plot(ge_attack)
    plt.xlabel('Number of Attack Traces')
    plt.ylabel('Guessing Entropy')
    plt.title('Desynchronization Nmax = 0')
    plt.show()

    # plt.plot(loss,label="loss")
    # plt.plot(val_loss,label="val_loss")
    # plt.legend(loc='upper right')
    # plt.ylabel('loss')
    # plt.xlabel('epochs')
    # plt.show()
    #
    #
    # plt.plot(accuracy,label="accuracy")
    # plt.plot(val_accuracy,label="val_accuracy")
    # plt.legend(loc='upper right')
    # plt.ylabel('accuracy')
    # plt.xlabel('epochs')
    # plt.show()
