# TensorFlow and tf.keras
import tensorflow as tf
#Helpoer libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import glob
import soundfile
from subscript import FFTTool, mellogspec

def ComplileModel(training_list, label_list):
    """
    Compile machine learning model.
    Here is DNN model, classify number from 0 to 9.

    Attribute:
        training_list: saved data training to deep learning.
        label_list: saved labels training to deep learning.

    Return:
        model: object of model using deep learning
        X_test: data of test to use get_accuracy().
        y_test: labels of test to use get_accuracy().
    """
    X_train, X_test, y_train, y_test = train_test_split(
        training_list, label_list, random_state=0
    )
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10)
    return model, X_test, y_test


def get_accuracy(model, X_test, y_test):
    """
    Output accuracy using test data and labels.

    Attributes:
        model: model of outputing CompileModel()
        X_test: data of test
        y_test: label of test

    Return:
        test_loss: loss value
        test_value: accuracy value
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)
    return test_loss, test_acc


def get_max_nframe(fpath):
    """
    The function of Decision of data length in wav files. It can get max data length from those wav files. 

    Attribute:
        fpath: file path of saving wave files

    Return:
        max_N: The longest data length of reading wav files.
    """
    files = glob.glob(fpath)
    N_list = []

    for fname in files:
        data, fs = soundfile.read(fname)
        N = len(data)
        N_list.append(N)
    
    max_N = max(N_list)
    return max_N


if __name__=="__main__":
    fpath = "recordings/*.wav"
    files = glob.glob(fpath)
    training_list = []
    label_list = []

    max_nframe = get_max_nframe(fpath)

    for fname in files:
        data, fs = soundfile.read(fname)
        data = FFTTool.ZeroPadding(data, max_nframe=max_nframe).process()
        window_data = np.hamming(len(data)) * data
        mellogspec_array, mel_scale = mellogspec.get_mellogspec(window_data, fs)
        mellogspec_array = preprocessing.scale(mellogspec_array)
        training_list.append(mellogspec_array)
        label = int(fname.split('/')[1].split('_')[0])
        label_list.append(label)

    #Deep learning
    compile_model, X_test, y_test = ComplileModel(training_list, label_list)
    get_accuracy(compile_model, X_test, y_test)
