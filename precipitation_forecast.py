SEED_VALUE = 100
import warnings
import os

os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
import random

random.seed(SEED_VALUE)
import numpy as np

np.random.seed(SEED_VALUE)
import pandas as pd
import tensorflow as tf

tf.random.set_seed(SEED_VALUE)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import InputLayer, SimpleRNN, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import RandomUniform, Ones, Zeros
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import EarlyStopping
from termcolor import colored
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot

pyplot.rcParams['figure.figsize'] = [10, 5]

warnings.filterwarnings("ignore")
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.get_logger().setLevel('ERROR')
print("TensorFlow version: {}".format(tf.__version__))
print("Keras version: {}".format(tf.keras.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


##########################################################################################
##########################################################################################
class Data:
    def __init__(self, years, features):
        self.years = years
        self.features = features
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.x = pd.DataFrame([], columns=self.features[:-1])
        self.y = pd.DataFrame([], columns=self.features[-1:])

    def load_dataset(self, file_name):
        dtypes = [float for i in range(len(self.features))]
        df = pd.DataFrame()
        for f, t in zip(self.features, dtypes):
            df[f] = pd.Series(dtype=t)
        df = pd.read_csv(file_name, header=0, usecols=self.features, dtype={self.features[i]: dtypes[i] for i in range(len(dtypes))})
        # remove rows with invalid PRCP (>99)
        df = df[df.keys()][df['PRCP'] < 99]
        # remove rows with invalid STP (<200)
        df = df[df.keys()][df['STP'] > 200]
        self.y = df[df.keys()[-1:]]
        self.x = df[df.keys()]
        print(colored('max PRCP = {}, min PRCP = {}'.format(max(self.y.values[:, 0]), min(self.y.values[:, 0])), 'magenta'))

    def clip_prcp(self, max_prcp):
        self.y.clip(lower=0, upper=max_prcp, inplace=True)
        self.x['PRCP'] = self.x['PRCP'].clip(lower=0, upper=max_prcp)

    def normalize(self, scaler_x=None, scaler_y=None):
        if scaler_x is None:
            self.scaler_x.fit(self.x.values)
            self.scaler_y.fit(self.y.values)
            norm_x = self.scaler_x.transform(self.x.values)
            norm_y = self.scaler_y.transform(self.y.values)
            self.x = pd.DataFrame(data=norm_x, columns=self.features)
            self.y = pd.DataFrame(data=norm_y, columns=self.features[-1:])
        else:
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y
            norm_x = self.scaler_x.transform(self.x.values)
            norm_y = self.scaler_y.transform(self.y.values)
            self.x = pd.DataFrame(data=norm_x, columns=self.features)
            self.y = pd.DataFrame(data=norm_y, columns=self.features[-1:])

    def series_to_sequences(self, shift_len_x, shift_len_y, skip_x_len):
        keys_x = self.x.keys()
        key_y = self.y.keys()
        cols_x, names_x = list(), list()
        cols_y, names_y = list(), list()
        # input fo coordinate features
        cols_x.append(self.x[keys_x[:skip_x_len]].reset_index(drop=True))
        names_x += [('{}'.format(keys_x[j])) for j in range(skip_x_len)]
        # input sequence (t0, ... tn)
        for t in range(shift_len_x, 0, -1):
            cols_x.append(self.x[keys_x[skip_x_len:]].shift(t).reset_index(drop=True))
            names_x += [('{}(t-{})'.format(keys_x[j], t)) for j in range(skip_x_len, len(keys_x))]
        # forecast sequence (t0, ... tn)
        for t in range(shift_len_y):
            cols_y.append(self.y[key_y[-1]].shift(-t).reset_index(drop=True))
            names_y += [('{}(t+{})'.format(key_y[-1], t))]
        # put it all together
        values_x = pd.concat(cols_x, axis=1, ignore_index=True).values
        values_y = pd.concat(cols_y, axis=1, ignore_index=True).values
        self.x = pd.DataFrame(data=values_x, columns=names_x)
        self.y = pd.DataFrame(data=values_y, columns=names_y)
        # drop rows with NaN values
        idx = self.x.isna().any(axis=1)
        self.x.drop(self.x.index[idx], inplace=True)
        self.y.drop(self.y.index[idx], inplace=True)

    def series_to_categories(self, bands):
        # convert y series into categories
        # only first column of y is used
        y = self.get_y()[:, 0]
        num_rows, num_cols = y.shape[0], len(bands) - 1
        y_categorical = np.zeros((num_rows, num_cols), dtype=int)
        for row in range(num_rows):
            for col in range(0, num_cols):
                if y[row] < bands[col + 1]:
                    y_categorical[row, col] = 1
                    break
        self.y = pd.DataFrame(data=y_categorical)

    def divide_train_validation(self, x, y, percentage=0.4):
        num_classes = y.shape[1]
        x_, y_ = np.empty((0, x.shape[1], x.shape[2])), np.empty((0, y.shape[1]))
        # select validation set from training set randomly
        for cls in range(num_classes):
            idx_cls = np.where(y[:, cls] == 1)[0]
            idx_valid = np.random.choice(idx_cls, int(percentage * idx_cls.shape[0]), replace=False)
            x_, y_ = np.append(x_, x[idx_valid, :, :], axis=0), np.append(y_, y[idx_valid, :], axis=0)
            x, y = np.delete(x, idx_valid, axis=0), np.delete(y, idx_valid, axis=0)
        return (x, y), (x_, y_)

    def get_x(self):
        return self.x.values

    def get_y(self):
        return self.y.values

    def get_scaler_x(self):
        return self.scaler_x

    def get_scaler_y(self):
        return self.scaler_y


class ModelRNN:
    def __init__(self):
        self.model = Sequential()

    def create_model(self, input_shape, output_dim):
        self.model.add(SimpleRNN(50,
                                 input_shape=input_shape,
                                 kernel_initializer=RandomUniform(seed=SEED_VALUE),
                                 bias_initializer=Zeros(),
                                 return_sequences=True,
                                 activation='relu'))
        self.model.add(SimpleRNN(50,
                                 kernel_initializer=RandomUniform(seed=SEED_VALUE),
                                 bias_initializer=Zeros(),
                                 activation='relu', ))
        # self.model.add(Dense(10,
        #                      kernel_initializer=RandomUniform(seed=SEED_VALUE),
        #                      bias_initializer=Zeros(),
        #                      activation='relu'))
        self.model.add(Dense(output_dim,
                             kernel_initializer=RandomUniform(seed=SEED_VALUE),
                             bias_initializer=Zeros(),
                             # use_bias=False,
                             activation='relu'))
        print(colored('Model RNN', 'green'))
        self.model.summary()

    def create_model_categorical(self, input_shape, output_dim):
        self.model.add(SimpleRNN(50,
                                 input_shape=input_shape,
                                 kernel_initializer=RandomUniform(seed=SEED_VALUE),
                                 bias_initializer=Zeros(),
                                 return_sequences=True,
                                 activation='relu'))
        self.model.add(SimpleRNN(50,
                                 kernel_initializer=RandomUniform(seed=SEED_VALUE),
                                 bias_initializer=Zeros(),
                                 # return_sequences=True,
                                 activation='relu'))
        # self.model.add(Dense(20,
        #                      kernel_initializer=RandomUniform(seed=SEED_VALUE),
        #                      bias_initializer=Zeros(),
        #                      activation='relu'))
        self.model.add(Dense(output_dim,
                             kernel_initializer=RandomUniform(seed=SEED_VALUE),
                             bias_initializer=Zeros(),
                             activation='softmax'))
        print(colored('Model RNN', 'green'))
        self.model.summary()

    def train_model(self, train_set, validation_set, epochs=50, lr=0.001, verbose=0):
        print(colored('Training ...', 'green'))
        self.model.compile(loss='mean_squared_error',
                           # optimizer=Adagrad(learning_rate=lr),
                           # optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                           optimizer=Adam(learning_rate=lr),
                           metrics=['mae'])
        his = self.model.fit(x=train_set[0],
                             y=train_set[1],
                             validation_data=validation_set,
                             epochs=epochs,
                             batch_size=128,
                             shuffle=True,
                             verbose=verbose)
        return his

    def train_model_categorical(self, train_set, validation_set, epochs=50, lr=0.001, verbose=0):
        print(colored('Training ...', 'green'))
        self.model.compile(loss='categorical_crossentropy',
                           # optimizer=Adagrad(learning_rate=lr),
                           # optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                           optimizer=Adam(learning_rate=lr),
                           metrics=['categorical_accuracy'])
        his = self.model.fit(x=train_set[0],
                             y=train_set[1],
                             validation_data=validation_set,
                             epochs=epochs,
                             batch_size=128,
                             shuffle=True,
                             verbose=verbose)
        return his

    def predict_model(self, test_set, scaler):
        print(colored('Testing ...', 'green'))
        y_ = self.model.predict(test_set[0])
        y_ = denormalize(y_, scaler)
        y_ = np.reshape(y_, (len(y_),))
        y = test_set[1]
        y = denormalize(y, scaler)
        y = np.reshape(y, (len(y),))
        return y, y_

    def predict_model_categorical(self, test_set):
        print(colored('Testing ...', 'green'))
        y_ = np.argmax(self.model.predict(test_set[0]), axis=1)
        y_ = np.reshape(y_, (len(y_),))
        y = np.argmax(test_set[1], axis=1)
        y = np.reshape(y, (len(y),))
        return y, y_

    def model_accuracy(self, y_true, y_pred):
        return np.mean(np.power(y_true, 2) - np.power(y_pred, 2))

    def model_accuracy_categorical(self, y_true, y_pred):
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        accuracy = sum(cm.diagonal()) / len(y_true)
        return cm, class_accuracy, accuracy

    def plot_training(self, history, file_name):
        if not os.path.exists('Results'):
            os.makedirs('Results')
        fig, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2)
        ax1.plot(history.epoch, history.history['loss'], color='b', label='Training')
        ax1.plot(history.epoch, history.history['val_loss'], color='r', label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax2.plot(history.epoch, history.history['mae'], color='b', label='Training')
        ax2.plot(history.epoch, history.history['val_mae'], color='r', label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1.05])
        ax2.legend(loc='lower right')
        pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.2)
        pyplot.savefig('Results\{}'.format(file_name), dpi=600)
        pyplot.show()

    def plot_training_categorical(self, history, file_name):
        if not os.path.exists('Results'):
            os.makedirs('Results')
        fig, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2)
        ax1.plot(history.epoch, history.history['loss'], color='b', label='Training')
        ax1.plot(history.epoch, history.history['val_loss'], color='r', label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax2.plot(history.epoch, history.history['categorical_accuracy'], color='b', label='Training')
        ax2.plot(history.epoch, history.history['val_categorical_accuracy'], color='r', label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1.05])
        ax2.legend(loc='lower right')
        pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.2)
        pyplot.savefig('Results\{}'.format(file_name), dpi=600)
        pyplot.show()

    def save_model(self, name):
        if not os.path.exists('Results'):
            os.makedirs('Results')
        self.model.save('Results\{}'.format(name))

    def load_model(self, name):
        self.model = tf.keras.models.load_model(name)


class ModelLSTM:
    def __init__(self):
        self.model = Sequential()

    def create_model(self, input_shape, output_dim):
        self.model.add(LSTM(50,
                            input_shape=input_shape,
                            kernel_initializer=RandomUniform(seed=SEED_VALUE),
                            bias_initializer=Zeros(),
                            return_sequences=True,
                            activation='relu'))
        self.model.add(LSTM(50,
                            kernel_initializer=RandomUniform(seed=SEED_VALUE),
                            bias_initializer=Zeros(),
                            activation='relu', ))
        # self.model.add(Dense(10,
        #                      kernel_initializer=RandomUniform(seed=SEED_VALUE),
        #                      bias_initializer=Zeros(),
        #                      activation='relu'))
        self.model.add(Dense(output_dim,
                             kernel_initializer=RandomUniform(seed=SEED_VALUE),
                             bias_initializer=Zeros(),
                             # use_bias=False,
                             activation='relu'))
        print(colored('Model RNN', 'green'))
        self.model.summary()

    def create_model_categorical(self, input_shape, output_dim):
        self.model.add(LSTM(50,
                            input_shape=input_shape,
                            kernel_initializer=RandomUniform(seed=SEED_VALUE),
                            bias_initializer=Zeros(),
                            return_sequences=True,
                            activation='relu'))
        self.model.add(LSTM(50,
                            kernel_initializer=RandomUniform(seed=SEED_VALUE),
                            bias_initializer=Zeros(),
                            # return_sequences=True,
                            activation='relu'))
        # self.model.add(Dense(20,
        #                      kernel_initializer=RandomUniform(seed=SEED_VALUE),
        #                      bias_initializer=Zeros(),
        #                      activation='relu'))
        self.model.add(Dense(output_dim,
                             kernel_initializer=RandomUniform(seed=SEED_VALUE),
                             bias_initializer=Zeros(),
                             activation='softmax'))
        print(colored('Model RNN', 'green'))
        self.model.summary()

    def train_model(self, train_set, validation_set, epochs=50, lr=0.001, verbose=0):
        print(colored('Training ...', 'green'))
        self.model.compile(loss='mean_squared_error',
                           # optimizer=Adagrad(learning_rate=lr),
                           # optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                           optimizer=Adam(learning_rate=lr),
                           metrics=['mae'])
        his = self.model.fit(x=train_set[0],
                             y=train_set[1],
                             validation_data=validation_set,
                             epochs=epochs,
                             batch_size=128,
                             shuffle=True,
                             verbose=verbose)
        return his

    def train_model_categorical(self, train_set, validation_set, epochs=50, lr=0.001, verbose=0):
        print(colored('Training ...', 'green'))
        self.model.compile(loss='categorical_crossentropy',
                           # optimizer=Adagrad(learning_rate=lr),
                           # optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                           optimizer=Adam(learning_rate=lr),
                           metrics=['categorical_accuracy'])
        his = self.model.fit(x=train_set[0],
                             y=train_set[1],
                             validation_data=validation_set,
                             epochs=epochs,
                             batch_size=128,
                             shuffle=True,
                             verbose=verbose)
        return his

    def predict_model(self, test_set, scaler):
        print(colored('Testing ...', 'green'))
        y_ = self.model.predict(test_set[0])
        y_ = denormalize(y_, scaler)
        y_ = np.reshape(y_, (len(y_),))
        y = test_set[1]
        y = denormalize(y, scaler)
        y = np.reshape(y, (len(y),))
        return y, y_

    def predict_model_categorical(self, test_set):
        print(colored('Testing ...', 'green'))
        y_ = np.argmax(self.model.predict(test_set[0]), axis=1)
        y_ = np.reshape(y_, (len(y_),))
        y = np.argmax(test_set[1], axis=1)
        y = np.reshape(y, (len(y),))
        return y, y_

    def model_accuracy(self, y_true, y_pred):
        return np.mean(np.power(y_true, 2) - np.power(y_pred, 2))

    def model_accuracy_categorical(self, y_true, y_pred):
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        accuracy = sum(cm.diagonal()) / len(y_true)
        return cm, class_accuracy, accuracy

    def plot_training(self, history, file_name):
        if not os.path.exists('Results'):
            os.makedirs('Results')
        fig, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2)
        ax1.plot(history.epoch, history.history['loss'], color='b', label='Training')
        ax1.plot(history.epoch, history.history['val_loss'], color='r', label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax2.plot(history.epoch, history.history['mae'], color='b', label='Training')
        ax2.plot(history.epoch, history.history['val_mae'], color='r', label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1.05])
        ax2.legend(loc='lower right')
        pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.2)
        pyplot.savefig('Results\{}'.format(file_name), dpi=600)
        pyplot.show()

    def plot_training_categorical(self, history, file_name):
        if not os.path.exists('Results'):
            os.makedirs('Results')
        fig, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2)
        ax1.plot(history.epoch, history.history['loss'], color='b', label='Training')
        ax1.plot(history.epoch, history.history['val_loss'], color='r', label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax2.plot(history.epoch, history.history['categorical_accuracy'], color='b', label='Training')
        ax2.plot(history.epoch, history.history['val_categorical_accuracy'], color='r', label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1.05])
        ax2.legend(loc='lower right')
        pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.2)
        pyplot.savefig('Results\{}'.format(file_name), dpi=600)
        pyplot.show()

    def save_model(self, name):
        if not os.path.exists('Results'):
            os.makedirs('Results')
        self.model.save('Results\{}'.format(name))

    def load_model(self, name):
        self.model = tf.keras.models.load_model(name)


class ModelAutoEncoder:
    def __init__(self):
        self.autoencoder = Sequential()

    def create_autoencoder(self, input_dim, encoding_dim):
        compression_factor = float(input_dim) / encoding_dim
        print("Compression factor: %s" % compression_factor)
        self.autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,), activation='sigmoid'))
        self.autoencoder.add(Dense(input_dim, activation='relu'))
        print(colored('AutoEncoder', 'green'))
        self.autoencoder.summary()

    def train_autoencoder(self, train_set, validation_set, epochs=50, lr=0.001, verbose=0, plot_training=True):
        self.autoencoder.compile(loss='mse',
                                 optimizer=Adam(learning_rate=lr))
        history = self.autoencoder.fit(x=train_set[0],
                                       y=train_set[1],
                                       validation_data=validation_set,
                                       epochs=epochs,
                                       batch_size=128,
                                       verbose=verbose,
                                       shuffle=True)
        if plot_training:
            pyplot.plot(history.history['loss'], label='Train')
            pyplot.plot(history.history['val_loss'], label='Validation')
            pyplot.xlabel('Epochs')
            pyplot.ylabel('Loss')
            pyplot.legend()
            pyplot.show()

    def get_encoder(self):
        encoder = Sequential()
        for layer in self.autoencoder.layers[:-1]:
            encoder.add(layer)
        return encoder


def oversample(x, y, sampling_strategy=None):
    if sampling_strategy is None:
        sampling_strategy = 'auto'
    oversampler = SMOTE(sampling_strategy=sampling_strategy, random_state=SEED_VALUE, k_neighbors=5)
    labels = np.argmax(y, axis=1)
    init_shape = x.shape
    x = np.reshape(x, (x.shape[0], -1))
    x, labels = oversampler.fit_resample(x, labels)
    x = np.clip(x, 0, 1)
    x = np.reshape(x, (x.shape[0], init_shape[1], init_shape[2]))
    y = np.zeros((labels.shape[0], y.shape[1]), dtype=int)
    y[np.arange(y.shape[0]), labels] = 1
    return x, y


def class_weight(y):
    labels = range(y.shape[1])
    y = np.argmax(y, axis=1)
    weights = np.zeros_like(labels, dtype=float)
    for i in labels:
        weights[i] = sum(y == labels[i]) / len(y)
    return weights


def denormalize(data, scaler):
    return scaler.inverse_transform(data)


##########################################################################################
##########################################################################################
if __name__ == '__main__':
    # selected features
    # FEATURES = ['LATITUDE',	'LONGITUDE', 'ELEVATION', 'TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP']
    FEATURES = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'TEMP', 'MAX', 'MIN', 'DEWP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'PRCP']
    # first COORDINATE_FEATURES_LEN features are coordinate features
    # last feature is the one that should be predicted
    COORDINATE_FEATURES_LEN = 3

    # selected years
    YEARS = range(2008, 2020)
    YEARS_TRAIN = YEARS[:10]
    YEARS_TEST = YEARS[10:]
    print(colored('years_to_train = ', 'yellow'), colored(['{:4d}'.format(y) for y in YEARS_TRAIN], 'yellow'))
    print(colored('years_to_test  = ', 'yellow'), colored(['{:4d}'.format(y) for y in YEARS_TEST], 'yellow'))

    # shift length
    X_SHIFT_LENGTH, Y_SHIFT_LENGTH = 3, 1

    # categorical prediction
    PRCP_BANDS = [0, 0.001, 0.2, 0.6, 1.2, 1.5]
    DESIRED_WEIGHTS = [1, 0.5, 0.25, 0.25, 0.125]

    # continents
    CONTINENT = ['europe', 'north_america', 'south_america']
    CONTINENT_TRAINING = CONTINENT[0]
    CONTINENT_TESTING = CONTINENT[0]
    print(colored('CONTINENT_TRAINING = {}'.format(CONTINENT_TRAINING), 'green'))
    print(colored('CONTINENT_TESTING  = {}'.format(CONTINENT_TESTING), 'green'))

    # current folder
    root_folder = os.getcwd()
    root_folder = os.path.join(root_folder, 'Files')
    print('root folder : ', root_folder)
    os.chdir(root_folder)

    # training data
    root_folder_training = os.path.join(root_folder, CONTINENT_TRAINING.lower())
    os.chdir(root_folder_training)
    print('root folder (training) : ', root_folder_training)
    data_train = Data(YEARS_TRAIN, FEATURES)
    # read csv file
    fn_train = 'dataset_train.csv'
    data_train.load_dataset(fn_train)
    data_train.clip_prcp(PRCP_BANDS[-1] - 0.001)
    # pyplot.hist(data_train.get_y()[:, 0], bins=PRCP_BANDS)
    # pyplot.show()
    # normalize data
    data_train.normalize()
    # shift samples
    data_train.series_to_sequences(X_SHIFT_LENGTH, Y_SHIFT_LENGTH, COORDINATE_FEATURES_LEN)
    # convert output from continuous variables to classes
    prcp_bands_normalized = data_train.get_scaler_y().transform(np.reshape(PRCP_BANDS, (len(PRCP_BANDS), 1)))
    prcp_bands_normalized = np.reshape(prcp_bands_normalized, (-1,))
    data_train.series_to_categories(prcp_bands_normalized)
    x_train, y_train = data_train.get_x(), data_train.get_y()
    x_train = x_train[:, COORDINATE_FEATURES_LEN:]
    x_train = x_train.reshape((x_train.shape[0], X_SHIFT_LENGTH, -1))
    print(colored('x_train.shape = {}'.format(x_train.shape), 'blue'))
    print(colored('y_train.shape = {}'.format(y_train.shape), 'blue'))
    (x_train, y_train), (x_valid, y_valid) = data_train.divide_train_validation(x_train, y_train, 0.3)
    print(colored('x_train.shape = {}'.format(x_train.shape), 'blue'))
    print(colored('y_train.shape = {}'.format(y_train.shape), 'blue'))
    print(colored('x_valid.shape = {}'.format(x_valid.shape), 'blue'))
    print(colored('y_valid.shape = {}'.format(y_valid.shape), 'blue'))
    weights_valid = class_weight(y_valid)
    print('validation weights                   : ', ['[{:2d} : {:.3f}]'.format(i, weights_valid[i]) for i in range(len(weights_valid))])
    weights_train = class_weight(y_train)
    print('training weights before oversampling : ', ['[{:2d} : {:.3f}]'.format(i, weights_train[i]) for i in range(len(weights_train))])
    # length of class 0 will be kept the same
    max_len = max(np.sum(y_train, axis=0))
    desired_len = [int(max_len * w) for w in DESIRED_WEIGHTS]
    strategy = dict(zip(range(y_train.shape[1]), desired_len))
    x_train, y_train = oversample(x_train, y_train, strategy)
    # x_train, y_train = oversample(x_train, y_train)
    weights_train = class_weight(y_train)
    print('training weights after oversampling  : ', ['[{:2d} : {:.3f}]'.format(i, weights_train[i]) for i in range(len(weights_train))])
    print(colored('x_train.shape = {}'.format(x_train.shape), 'blue'))
    print(colored('y_train.shape = {}'.format(y_train.shape), 'blue'))

    # testing data)
    root_folder_testing = os.path.join(root_folder, CONTINENT_TESTING.lower())
    os.chdir(root_folder_testing)
    print('root folder (testing)  : ', root_folder_testing)
    data_test = Data(YEARS_TEST, FEATURES)
    # read csv file
    fn_test = 'dataset_test.csv'
    data_test.load_dataset(fn_test)
    data_test.clip_prcp(PRCP_BANDS[-1] - 0.001)
    # normalize data
    data_test.normalize(data_train.get_scaler_x(), data_train.get_scaler_y())
    # shift samples
    data_test.series_to_sequences(X_SHIFT_LENGTH, Y_SHIFT_LENGTH, COORDINATE_FEATURES_LEN)
    # convert output from continuous variables to classes
    data_test.series_to_categories(prcp_bands_normalized)
    x_test, y_test = data_test.get_x(), data_test.get_y()
    x_test = x_test[:, COORDINATE_FEATURES_LEN:]
    x_test = x_test.reshape((x_test.shape[0], X_SHIFT_LENGTH, -1))
    print(colored('x_test.shape  = {}'.format(x_test.shape), 'blue'))
    print(colored('y_test.shape  = {}'.format(y_test.shape), 'blue'))
    weights_test = class_weight(y_test)
    print('test weights                         : ', ['[{:2d} : {:.3f}]'.format(i, weights_test[i]) for i in range(len(weights_test))])

    # go back to the root folder
    os.chdir(root_folder)

    # model (Categorical)
    # model = ModelRNN()
    model = ModelLSTM()
    model.create_model_categorical(input_shape=(x_train.shape[1], x_train.shape[2]), output_dim=y_train.shape[1])
    history = model.train_model_categorical((x_train, y_train), (x_valid, y_valid), epochs=50, lr=0.001, verbose=2)
    model.save_model('{}.h5'.format(CONTINENT_TRAINING.upper()))
    model.plot_training_categorical(history, '{}.png'.format(CONTINENT_TRAINING.upper()))
    # model.load_model('{}.h5'.format(CONTINENT_TRAINING.upper()))
    # model performance on the validation set
    y_true, y_pred = model.predict_model_categorical((x_valid, y_valid))
    cm, class_accuracy, accuracy = model.model_accuracy_categorical(y_true=y_true, y_pred=y_pred)
    print(colored('Validation Results', 'yellow'))
    print(colored('Confusion Matrix', 'green'))
    print(cm)
    print(colored('Classification Report', 'green'))
    print('accuracy = {:.3f}'.format(accuracy))
    print('class accuracy = ')
    for i in range(len(class_accuracy)):
        print('\tclass {} : {:.3f}'.format(i, class_accuracy[i]))
    # model performance on the Testing set
    y_true, y_pred = model.predict_model_categorical((x_test, y_test))
    y_df = pd.DataFrame({'TRUE': y_true, 'PRED': y_pred}, index=range(np.shape(y_true)[0]))
    if not os.path.exists('Results'):
        os.makedirs('Results')
    y_df.to_csv('Results/test_vs_pred_{}_{}.csv'.format(CONTINENT_TRAINING, CONTINENT_TESTING), index=False)
    cm, class_accuracy, accuracy = model.model_accuracy_categorical(y_true=y_true, y_pred=y_pred)
    print(colored('Testing Results', 'yellow'))
    print(colored('Confusion Matrix', 'green'))
    print(cm)
    print(colored('Classification Report', 'green'))
    print('accuracy = {:.3f}'.format(accuracy))
    print('class accuracy = ')
    for i in range(len(class_accuracy)):
        print('\tclass {} : {:.3f}'.format(i, class_accuracy[i]))

    # # model (Regression)
    # model = ModelRNN()
    # model.create_model(input_shape=(x_train.shape[1], x_train.shape[2]), output_dim=y_train.shape[1])
    # history = model.train_model((x_train, y_train), (x_valid, y_valid), epochs=50, lr=0.001, verbose=2)
    # model.save_model('{}.h5'.format(CONTINENT_TRAINING.upper()))
    # model.plot_training(history, '{}.png'.format(CONTINENT_TRAINING.upper()))
    # # model.load_model('{}.h5'.format(CONTINENT_TRAINING.upper()))
    # # model performance on the validation set
    # y_true, y_pred = model.predict_model_categorical((x_valid, y_valid))
    # accuracy = model.model_accuracy(y_true=y_true, y_pred=y_pred)
    # print(colored('Validation Results', 'yellow'))
    # print(colored('Classification Report', 'green'))
    # print('accuracy = {:.3f}'.format(accuracy))
    # # model performance on the Testing set
    # y_true, y_pred = model.predict_model_categorical((x_test, y_test))
    # y_df = pd.DataFrame({'TRUE': y_true, 'PRED': y_pred}, index=range(np.shape(y_true)[0]))
    # if not os.path.exists('Results'):
    #     os.makedirs('Results')
    # y_df.to_csv('Results/test_vs_pred_{}_{}.csv'.format(CONTINENT_TRAINING, CONTINENT_TESTING), index=False)
    # accuracy = model.model_accuracy(y_true=y_true, y_pred=y_pred)
    # print(colored('Testing Results', 'yellow'))
    # print(colored('Classification Report', 'green'))
    # print('accuracy = {:.3f}'.format(accuracy))

    # autoencoder
    # x_train_reshaped = x_train.reshape((x_train.shape[0], -1))
    # x_valid_reshaped = x_valid.reshape((x_valid.shape[0], -1))
    # autoencoder = ModelAutoEncoder()
    # autoencoder.create_autoencoder(x_train_reshaped.shape[1], encoding_dim=10)
    # autoencoder.train_autoencoder((x_train_reshaped, x_train_reshaped), (x_valid_reshaped, x_valid_reshaped), epochs=20, lr=0.001, verbose=2, plot_training=True)
    # encoder = autoencoder.get_encoder()
    # encoder.summary()
