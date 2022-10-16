import argparse
import logging
import os
import datetime

from tensorflow import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import reuters
from keras import models, layers
from keras.utils.np_utils import to_categorical
from tensorflow.keras import regularizers,optimizers

import numpy as np

logger = logging.getLogger(__name__)


class NewsWire:
    @staticmethod
    def feed_forward(hidden_unit):
        model = models.Sequential()
        # model.add(layers.Dropout(0.5, input_shape=(10000,)))
        model.add(layers.Dense(hidden_unit, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(10000,)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(hidden_unit, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(46, activation='softmax'))
        opt = optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def train(model, x_train, y_train, epoch, batch_size):
        # x_val = x_train[:1000]
        # partial_x_train = x_train[1000:]
        # y_val = y_train[:1000]
        # partial_y_train = y_train[1000:]
        tb_path = os.path.join(f'tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

        os.makedirs(tb_path, exist_ok=True)
        tensorboard = TensorBoard(log_dir=tb_path)
        cp_path = 'checkpoints'
        os.makedirs(cp_path, exist_ok=True)
        cp_callback = ModelCheckpoint(filepath=os.path.join(cp_path, 'model.hdf5'),
                                      monitor='val_accuracy',
                                      save_freq='epoch', verbose=1, period=1, save_best_only=False)
        # model.fit(partial_x_train, partial_y_train, epochs=epoch, batch_size=batch_size, callbacks=[tensorboard, cp_callback], validation_data=(x_val, y_val) )
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, callbacks=[tensorboard, cp_callback])

    @staticmethod
    def evaluate(model, x_test, y_test):
        result = model.evaluate(x_test, y_test)
        return result

    @staticmethod
    def vectorize(sequences, dimension=10000):
        result = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            result[i, sequence] = 1
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=False, default=10,
                        help='configuration yaml file')
    parser.add_argument('--batch_size', type=int, required=False, default=512,
                        help='configuration yaml file')
    parser.add_argument('--hidden_unit', type=int, required=False, default=64,
                        help='configuration yaml file')
    args = parser.parse_args()
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    news = NewsWire()
    x_train = news.vectorize(train_data)
    x_test = news.vectorize(test_data)
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    # build the model
    model = news.feed_forward(args.hidden_unit)
    # train the model
    news.train(model, x_train, y_train, args.epoch, args.batch_size)
    # test the model
    print(f'The results of loss and accuracy  {news.evaluate(model, x_test, y_test)}')


if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        logger.exception("SystemExit or KeyboardInterrupt exception has caused application to terminate.")
    except Exception:
        logger.exception("Critical exception has caused application to terminate.")

[1.0724836587905884, 0.767141580581665]

[1.4261486530303955, 0.7960819005966187]