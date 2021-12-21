import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer



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

def create_cnn_x(width, height, depth, filters=(16, 32, 64), summarize=False, regress=False):

    inputs = Input(shape=(height, width, depth))

    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    if summarize:
        print(model.summarize())

    return model

def preprocess_images(df):
    train, test = (df['train_test'] == i for i in ['train', 'test'])
    x_train = np.stack(df[train]['image'])
    y_train = to_categorical(np.stack(df[train]['mnist_label']))
    x_test = np.stack(df[test]['image'])
    y_test = to_categorical(np.stack(df[test]['mnist_label']))
    return x_train, y_train, x_test, y_test


def preprocess_features(df, continuous, categorical):
    train, test = (df['train_test'] == i for i in ['train', 'test'])

    cs = MinMaxScaler().fit(df[train][continuous])
    train_continuous = cs.transform(df[train][continuous])
    test_continuous = cs.transform(df[test][continuous])

    if categorical:
        binarizers = dict((feature, LabelBinarizer().fit(df[train][feature])) for feature in categorical)
        train_categorical = np.hstack([binarizers[feature].transform(df[train][feature]) for feature in categorical])
        test_categorical = np.hstack([binarizers[feature].transform(df[test][feature]) for feature in categorical])
    else:
        train_categorical = np.empty((len(df[train]), 0))
        test_categorical = np.empty((len(df[test]), 0))

    x_train = np.hstack([train_continuous, train_categorical])
    y_train = df[train]['shipping_fee'].values

    x_test = np.hstack([test_continuous, test_categorical])
    y_test = df[test]['shipping_fee'].values
    return x_train, y_train, x_test, y_test


def create_mlp(num_features, layer_sizes=[32, 16], regress=False, summarize=False):
    model = Sequential()
    model.add(Dense(layer_sizes[0], input_shape=(num_features,), activation='relu'))
    for layer_size in layer_sizes[1:]:
        model.add(Dense(layer_size, activation='relu'))

    if regress:
        model.add(Dense(1, activation='linear'))

    if summarize:
        print(model.summary())
    return model

def create_mlp(num_features, layer_sizes=[32, 16], regress=False, summarize=False):
    model = Sequential()
    model.add(Dense(layer_sizes[0], input_shape=(num_features,), activation='relu'))
    for layer_size in layer_sizes[1:]:
        model.add(Dense(layer_size, activation='relu'))

    if regress:
        model.add(Dense(1, activation='linear'))

    if summarize:
        print(model.summary())
    return model
