import argparse
import logging
import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import boston_housing
from keras import models, layers
from keras.utils.np_utils import to_categorical

import numpy as np

logger = logging.getLogger(__name__)


class HousePricePrediction:
    @staticmethod
    def feed_forward(hidden_unit, train_data):
        model = models.Sequential()
        model.add(layers.Dense(hidden_unit, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(hidden_unit, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def train(model, x_train, y_train, epoch, batch_size):
        tb_path = os.path.join('tensorboard/')

        os.makedirs(tb_path, exist_ok=True)
        tensorboard = TensorBoard(log_dir=tb_path)
        cp_path = 'checkpoints'
        os.makedirs(cp_path, exist_ok=True)
        cp_callback = ModelCheckpoint(filepath=os.path.join(cp_path, 'model.hdf5'),
                                      monitor='val_accuracy',
                                      save_freq='epoch', verbose=1, period=1, save_best_only=False)
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, callbacks=[tensorboard, cp_callback], )

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
    parser.add_argument('--epoch', type=int, required=False, default=20,
                        help='configuration yaml file')
    parser.add_argument('--batch_size', type=int, required=False, default=512,
                        help='configuration yaml file')
    parser.add_argument('--hidden_unit', type=int, required=False, default=64,
                        help='configuration yaml file')
    args = parser.parse_args()
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data(num_words=10000)
    price = HousePricePrediction()
    x_train = price.vectorize(train_data)
    x_test = price.vectorize(test_data)
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    # build the model
    model = price.feed_forward(args.hidden_unit, train_data)
    # train the model
    price.train(model, x_train, y_train, args.epoch, args.batch_size)
    # test the model
    print(f'The results of loss and accuracy  {price.evaluate(model, x_test, y_test)}')


if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        logger.exception("SystemExit or KeyboardInterrupt exception has caused application to terminate.")
    except Exception:
        logger.exception("Critical exception has caused application to terminate.")
