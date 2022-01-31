import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


def get_mnist_data():
    path = 'mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return x_train, y_train, x_test, y_test


def normalize(x):
    return tf.keras.utils.normalize(x, axis=1)


def build_optimal_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def optimal_model(x_train, y_train, x_test, y_test):
    model = build_optimal_model()
    train_begin_time = time.time()
    history = model.fit(x_train, y_train, epochs=6)
    train_end_time = time.time()
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print('val_loss:', val_loss)
    print('val_acc:', val_acc)
    print('train_time:', train_end_time - train_begin_time)
    epochs = [i + 1 for i in range(6)]
    plt.title('Training Monitoring')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Accuracy')
    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['accuracy'])
    plt.legend(['Train_Loss', 'Train_Accuracy'])
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_mnist_data()
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    optimal_model(x_train, y_train, x_test, y_test)
