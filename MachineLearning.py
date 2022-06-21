import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
import glob
import soundfile
from subscript import FFTTool, mellogspec

def ComplileModel(X_train, y_train):
    """
    Compile machine learning model.
    Here is DNN model, classify number from 0 to 9.

    Attribute:
        X_train: saved data training to deep learning.
        y_train: saved labels training to deep learning.

    Return:
        model: object of model using deep learning
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10)
    return model


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


    """Pre-processing"""
    max_nframe = get_max_nframe(fpath)
    N = len(files)
    training_list = []
    label_list = np.array([])

    for fname in files:
        data, fs = soundfile.read(fname)
        data = FFTTool.ZeroPadding(data, max_nframe=max_nframe).process()
        window_data = np.hamming(len(data)) * data
        mellogspec_array, mel_scale = mellogspec.get_mellogspec(window_data, fs) #generate data
        mellogspec_array = preprocessing.scale(mellogspec_array) #normalization
        training_list.append(mellogspec_array)
        label = int(fname.split('/')[1].split('_')[0])
        label_list = np.append(label_list, label)


    """Deep Learning"""
    from sklearn.model_selection import KFold
    loss_list = np.array([])
    acc_list = np.array([])
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, val_index in kf.split(training_list, label_list):
        train_data = np.array([training_list[i] for i in train_index])
        train_label = np.array([label_list[i] for i in train_index])
        val_data = np.array([training_list[i] for i in val_index])
        val_label = np.array([label_list[i] for i in val_index])
        model = ComplileModel(train_data, train_label)
        test_loss, test_acc = get_accuracy(model, val_data, val_label)
        loss_list = np.append(loss_list, test_loss)
        acc_list = np.append(acc_list, test_acc)

    print("Cross-validation loss: {}".format(np.mean(loss_list)))
    print("Cross-validation Accuracy: {}".format(np.mean(acc_list)))