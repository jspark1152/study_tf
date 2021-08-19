from numpy.core.fromnumeric import shape
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

nb_classes=10

mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)=mnist.load_data()


x_train = x_train/255
x_train = np.reshape(x_train, [-1, 784])
#x_train = tf.cast(x_train, tf.float32)

x_test = x_test / 255
x_test = tf.reshape(x_test, [-1, 784])

#y_train = tf.one_hot(y_train, nb_classes)
#y_train = tf.reshape(y_train, [-1, nb_classes])
y_test = tf.one_hot(y_test, nb_classes)
y_test = tf.reshape(y_test, [-1, nb_classes])

y_train=tf.keras.utils.to_categorical(y_train)

print(shape(y_train))

