
'''
Shallow Network Approximation Challenge

change the 'train_network' function to get a better approximation

read more at https://github.com/sukiboo/nn_approximation_challenge
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace')
np.random.seed(0)
tf.random.set_seed(0)


def train_network(x_train, y_train):
    '''create and train shallow network'''

    # create shallow network
    model = tf.keras.Sequential([tf.keras.Input(shape=(1,)),
                                 tf.keras.layers.Dense(10000, activation='relu'),
                                 tf.keras.layers.Dense(1, activation=None)])

    # select optimization algorithm
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # compile and train the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100000)

    return model


if __name__ == '__main__':

    # load training and test data
    (x_train, y_train), (x_test, y_test) = np.load('./data.npy', allow_pickle=True)

    # obtain the network
    model = train_network(x_train, y_train)

    # evaluate the model
    print('\nmodel evaluation:')
    model.evaluate(x_test, y_test, batch_size=x_test.size)

    # plot the result
    plt.figure(figsize=(8,5))
    plt.scatter(x_train, y_train, label='train data')
    plt.scatter(x_test, y_test, label='test data')
    plt.plot(x_train, model.predict(x_train).flatten(), linewidth=3, color='turquoise', label='network')
    plt.legend()
    plt.show()
