import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Reshape, Input, Activation, \
    MaxPooling2D, concatenate
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


def create_cnn_sequential(regress=False, summarize=False):
    # This is the old way of creating the model, which will not work when we combine with the other branch.
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

def create_cnn(input_shape, filters=(16, 32, 64), summarize=False, regress=False):

    inputs = Input(shape=input_shape)

    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    if regress:
        x = Dense(10, activation="softmax")(x)

    model = Model(inputs, x)

    if summarize:
        print(model.summary())

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

def preprocess(df, continuous, categorical=None):
    images_train, _, images_test, _ = preprocess_images(df)
    features_train, y_train, features_test, y_test = preprocess_features(df, continuous=continuous,
                                                                         categorical=categorical)
    x_train = [images_train, features_train]
    x_test = [images_test, features_test]
    return x_train, y_train, x_test, y_test


def create_mlp_sequential(num_features, layer_sizes=[32, 16], regress=False, summarize=False):
    # This is the old way of creating the model, which will not work when we combine with the other branch.
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
    inputs = Input(num_features,)

    for i, layer_size in enumerate(layer_sizes):
        if i == 0:
            x = inputs
        x = Dense(layer_size, activation='relu')(x)

    if regress:
        x = Dense(1, activation='linear')(x)

    model = Model(inputs, x)

    if summarize:
        print(model.summary())
    return model


def create_combined_model(image_shape, num_features, mlp_layer_sizes=[32, 32], combined_layer_sizes=[32, 32]):
    cnn = create_cnn(input_shape=image_shape)
    mlp = create_mlp(num_features=num_features, layer_sizes=mlp_layer_sizes)

    combined_output = concatenate([cnn.output, mlp.output])

    for i, layer_size in enumerate(combined_layer_sizes):
        if i == 0:
            x = combined_output
        x = Dense(4, activation="relu")(x)

    x = Dense(1, activation="linear")(x)

    model = Model(inputs=[cnn.input, mlp.input], outputs=x)

    return model