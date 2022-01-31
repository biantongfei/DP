import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


def get_mnist_data():
    path = '../mnist.npz'
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
    plt.xlabel(title)
    plt.ylabel('Loss')
    plt.plot(x, loss)
    plt.show()
    plt.close()

    plt.title('Accuracy of different ' + title)
    plt.xlabel(title)
    plt.ylabel('Accuracy')
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
    val_time = [i * 1000 for i in val_time]
    plt.ylabel('Validation Time (ms)')
    plt.plot(x, val_time)
    plt.show()
    plt.close()


def build_modelB():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def monitoring(x_train, y_train, x_test, y_test):
    model = build_modelB()
    epochs = 3
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


def filter_dimensionality_estimate(x_train, y_train, x_test, y_test):
    filter = [8, 16, 32, 64, 128]
    filter_list = []
    for f1 in filter:
        for f2 in filter:
            if f2 < f1:
                continue
            for f3 in filter:
                if f3 < f2:
                    continue
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Conv2D(f1, (3, 3), activation='relu', input_shape=(28, 28, 1)))
                model.add(tf.keras.layers.Conv2D(f2, (3, 3), activation='relu'))
                model.add(tf.keras.layers.MaxPooling2D(2, 2))
                model.add(tf.keras.layers.Conv2D(f3, (3, 3), activation='relu'))
                model.add(tf.keras.layers.MaxPooling2D(2, 2))
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(128, activation='relu'))
                model.add(tf.keras.layers.Dense(10, activation='softmax'))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                train_begin_time = time.time()
                model.fit(x_train, y_train, epochs=3, verbose=0)
                train_end_time = time.time()
                val_loss, val_acc = model.evaluate(x_test, y_test)
                val_end_time = time.time()
                filter_key = str(f1) + ',' + str(f2) + ',' + str(f3)
                filter_list.append({'filter_key': filter_key, 'loss': val_loss, 'acc': val_acc,
                                    'train_time': train_end_time - train_begin_time,
                                    'val_time': (val_end_time - train_end_time) / 313})
                print(f1, f2, f3)
    filter_top_7 = sorted(filter_list, key=lambda k: k['acc'], reverse=True)[:7]
    filter_name = [i['filter_key'] for i in filter_top_7]
    acc = [i['acc'] for i in filter_top_7]
    loss = [i['loss'] for i in filter_top_7]
    train_time = [i['train_time'] for i in filter_top_7]
    val_time = [i['val_time'] for i in filter_top_7]
    draw_plt(filter_name, loss, acc, train_time, val_time, 'Filter Dimensionality')


def kernel_size_estimate(x_train, y_train, x_test, y_test):
    size = [(3, 3), (5, 5), (7, 7)]
    loss = []
    acc = []
    train_time = []
    val_time = []
    for s in size:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(8, s, activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.Conv2D(64, s, activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Conv2D(64, s, activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=3, verbose=0)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss.append(val_loss)
        acc.append(val_acc)
        train_time.append(train_end_time - train_begin_time)
        val_time.append((val_end_time - train_end_time) / 313)
    size = [str(i) for i in size]
    draw_plt(size, loss, acc, train_time, val_time, 'Kernel Size')


def epochs_estimate(x_train, y_train, x_test, y_test):
    epochs = [2, 3, 4, 5, 6]
    loss = []
    acc = []
    train_time = []
    val_time = []
    for e in epochs:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=e, verbose=0)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss.append(val_loss)
        acc.append(val_acc)
        train_time.append(train_end_time - train_begin_time)
        val_time.append((val_end_time - train_end_time) / 313)
    epochs = [str(i) for i in epochs]
    draw_plt(epochs, loss, acc, train_time, val_time, 'Epochs')


def strides_estimate(x_train, y_train, x_test, y_test):
    strides = [1, 2]
    loss = []
    acc = []
    train_time = []
    val_time = []
    for s in strides:
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1), strides=s, padding='same'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=s, padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=s, padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=3, verbose=0)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss.append(val_loss)
        acc.append(val_acc)
        train_time.append(train_end_time - train_begin_time)
        val_time.append((val_end_time - train_end_time) / 313)
    strides = [str(i) for i in strides]
    print(strides[0], loss[0], acc[0], train_time[0], val_time[0])
    print(strides[1], loss[1], acc[1], train_time[1], val_time[1])


def padding_estimate(x_train, y_train, x_test, y_test):
    padding = [True, False]
    loss = []
    acc = []
    train_time = []
    val_time = []
    for p in padding:
        model = tf.keras.models.Sequential()
        if p:
            model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(tf.keras.layers.MaxPooling2D(2, 2))
            model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        else:
            model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='valid'))
            model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='valid'))
            model.add(tf.keras.layers.MaxPooling2D(2, 2))
            model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid'))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_begin_time = time.time()
        model.fit(x_train, y_train, epochs=3, verbose=0)
        train_end_time = time.time()
        val_loss, val_acc = model.evaluate(x_test, y_test)
        val_end_time = time.time()
        loss.append(val_loss)
        acc.append(val_acc)
        train_time.append(train_end_time - train_begin_time)
        val_time.append((val_end_time - train_end_time) / 313)
    padding = [str(i) for i in padding]
    print(padding[0], loss[0], acc[0], train_time[0], val_time[0])
    print(padding[1], loss[1], acc[1], train_time[1], val_time[1])


def optimal_model(x_train, y_train, x_test, y_test):
    model = build_modelB()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_begin_time = time.time()
    model.fit(x_train, y_train, epochs=3)
    train_end_time = time.time()
    val_loss, val_acc = model.evaluate(x_test, y_test)
    val_end_time = time.time()
    print(val_loss, val_acc, train_end_time - train_begin_time, (val_end_time - train_end_time) / 313)
    model.save('CNN_model.h5')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_mnist_data()
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    # optimal_model(x_train, y_train, x_test, y_test)
    # filter_dimensionality_estimate(x_train, y_train, x_test, y_test)
    # kernel_size_estimate(x_train, y_train, x_test, y_test)
    # epochs_estimate(x_train, y_train, x_test, y_test)
    # strides_estimate(x_train, y_train, x_test, y_test)
    # padding_estimate(x_train, y_train, x_test, y_test)
    monitoring(x_train, y_train, x_test, y_test)
