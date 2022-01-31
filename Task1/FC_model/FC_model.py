import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


def get_mnist_data():
    path = 'Task1/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return x_train, y_train, x_test, y_test


def normalize(x):
    return tf.keras.utils.normalize(x, axis=1)


def draw_plt(x, loss, acc, train_time, val_time, title):
    print('plotting')
    plt.title('Loss of different ' + title)
    plt.xlabel('Loss')
    plt.ylabel(title)
    plt.plot(x, loss)
    plt.show()
    plt.close()

    plt.title('Accuracy of different ' + title)
    plt.xlabel('Accuracy')
    plt.ylabel(title)
    plt.plot(x, acc)
    plt.show()
    plt.close()

    plt.title('Training Time of different ' + title)
    plt.xlabel(title)
    plt.ylabel('Training Time (s)')
    plt.plot(x, train_time)
    plt.show()
    plt.close()

    plt.title('Validation Time of different ' + title)
    plt.xlabel(title)
    plt.ylabel('Validation Time (ms)')
    val_time = [i * 1000 for i in val_time]
    plt.plot(x, val_time)
    plt.show()
    plt.close()


def build_modelA():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def monitoring(x_train, y_train, x_test, y_test):
    model = build_modelA()
    epochs = 5
    batch_size = 32
    steps_per_epoch = x_train.shape[0] / batch_size
    loss = []
    acc = []
    v_loss = []
    v_acc = []
    for e in range(epochs):
        for s in range(int(steps_per_epoch)):
            x = x_train[s * batch_size:(s + 1) * batch_size]
            y = y_train[s * batch_size:(s + 1) * batch_size]
            history = model.fit(x, y, epochs=1, steps_per_epoch=1, verbose=0)

            if s % 300 == 0:
                val_loss, val_acc = model.evaluate(x_test, y_test)
                loss.append(history.history['loss'][0])
                acc.append(history.history['accuracy'][0])
                v_loss.append(val_loss)
                v_acc.append(val_acc)
            print(e, s)

    tra_x = [(i + 1) * 300 for i in range(len(loss))]
    val_x = [(i + 1) * 300 for i in range(len(v_loss))]
    plt.title('Training Monitoring')
    plt.xlabel('Iteration')
    plt.ylabel('Loss and Accuracy')
    plt.plot(tra_x, acc)
    plt.plot(val_x, v_acc)
    plt.plot(tra_x, loss)
    plt.plot(val_x, v_loss)
    plt.legend(['Tra_Accuracy', 'Val_Accuracy', 'Tra_Loss', 'Val_Loss'])
    plt.show()


def optimizer_estimate(x_train, y_train, x_test, y_test):
    optimizer_list = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam']
    loss_list = []
    acc_list = []
    train_time = []
    val_time = []
    for optimizer in optimizer_list:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=6)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss_list.append(val_loss)
        acc_list.append(val_acc)
        train_time.append(train_end_time - train_begin_time)
        val_time.append((val_end_time - train_end_time) / 313)
    draw_plt(optimizer_list, loss_list, acc_list, train_time, val_time, 'Optimizer')


def epoch_estimate(x_train, y_train, x_test, y_test):
    epoch_list = [i + 1 for i in range(15)]
    loss_list = []
    acc_list = []
    train_time_list = []
    val_time_list = []
    for epoch in epoch_list:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=epoch)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss_list.append(val_loss)
        acc_list.append(val_acc)
        train_time_list.append(train_end_time - train_begin_time)
        val_time_list.append((val_end_time - train_end_time) / 313)
    epoch_list = [str(i) for i in epoch_list]
    draw_plt(epoch_list, loss_list, acc_list, train_time_list, val_time_list, 'Epochs')


def batchsize_estimate(x_train, y_train, x_test, y_test):
    batchsize_list = [10, 16, 32, 50, 100, 200, 600, 1000, 5000, 10000]
    loss_list = []
    acc_list = []
    train_time_list = []
    val_time_list = []
    for batchsize in batchsize_list:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=5, batch_size=batchsize)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss_list.append(val_loss)
        acc_list.append(val_acc)
        train_time_list.append(train_end_time - train_begin_time)
        val_time_list.append((val_end_time - train_end_time) / 313)
    batchsize_list = [str(i) for i in batchsize_list]
    draw_plt(batchsize_list, loss_list, acc_list, train_time_list, val_time_list, 'Batch Size')


def neuron_estimate(x_train, y_train, x_test, y_test):
    neuron_list = [10, 16, 32, 64, 128, 256, 512]
    loss_list = []
    acc_list = []
    train_time_list = []
    val_time_list = []
    for neuron in neuron_list:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(neuron, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(neuron, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=6)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss_list.append(val_loss)
        acc_list.append(val_acc)
        train_time_list.append(train_end_time - train_begin_time)
        val_time_list.append((val_end_time - train_end_time) / 313)
    neuron_list = [str(i) for i in neuron_list]
    draw_plt(neuron_list, loss_list, acc_list, train_time_list, val_time_list, 'Number of Neurons')


def layer_estimate(x_train, y_train, x_test, y_test):
    layer_list = [i + 1 for i in range(5)]
    loss_list = []
    acc_list = []
    train_time_list = []
    val_time_list = []
    for layer in layer_list:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        i = 0
        while i < layer:
            model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
            i += 1
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=6)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss_list.append(val_loss)
        acc_list.append(val_acc)
        train_time_list.append(train_end_time - train_begin_time)
        val_time_list.append((val_end_time - train_end_time) / 313)
    layer_list = [str(i) for i in layer_list]
    draw_plt(layer_list, loss_list, acc_list, train_time_list, val_time_list, 'Number of layers')


def optimal_model(x_train, y_train, x_test, y_test):
    model = build_modelA()
    train_begin_time = time.time()
    history = model.fit(x_train, y_train, epochs=5)
    train_end_time = time.time()
    val_loss, val_acc = model.evaluate(x_test, y_test)
    val_end_time = time.time()
    print(val_loss, val_acc, train_end_time - train_begin_time, (val_end_time - train_end_time) / 313)
    print(history.history)
    model.save('FC_model.h5')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_mnist_data()
    # print(x_train.shape, x_test.shape)
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    # monitoring(x_train, y_train, x_test, y_test)
    # optimizer_estimate(x_train, y_train, x_test, y_test)
    # neuron_estimate(x_train, y_train, x_test, y_test)
    # layer_estimate(x_train, y_train, x_test, y_test)
    # epoch_estimate(x_train, y_train, x_test, y_test)
    # batchsize_estimate(x_train, y_train, x_test, y_test)
    optimal_model(x_train, y_train, x_test, y_test)
