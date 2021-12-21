import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape


def create_cnn(regress=False, summarize=False):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    if regress:
        model.add(Dense(10, activation='softmax'))

    if summarize:
        print(model.summary())

    return model


def create_mlp(regress=False, summarize=False):
    