import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

nb_classes=10

mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)=mnist.load_data()


x_train = x_train / 255
x_train = np.reshape(x_train, [-1, 784])
#x_train = tf.cast(x_train, tf.float32)
x_test = x_test / 255
x_test = np.reshape(x_test, [-1, 784])
#x_test = tf.cast(x_test, tf.float32)


y_train=tf.keras.utils.to_categorical(y_train)

y_test=tf.keras.utils.to_categorical(y_test)
#y_train = tf.one_hot(y_train, nb_classes)
#y_train = tf.reshape(y_train, [-1, nb_classes])
#y_test = tf.one_hot(y_test, nb_classes)
#y_test = tf.reshape(y_test, [-1, nb_classes])


X=tf.placeholder(tf.float32, shape=[None,784])
Y=tf.placeholder(tf.float32, shape=[None,nb_classes])


W=tf.Variable(tf.random_normal([784,nb_classes]), name='weight')
b=tf.Variable(tf.random_normal([1,nb_classes]), name='bias')

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

cost=tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypothesis), axis=1))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction=tf.argmax(hypothesis,1)
correction=tf.cast(tf.equal(prediction, tf.argmax(Y, 1)), tf.float32)
accuracy=tf.reduce_mean(correction)

batch_size=100
epochs=30

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        batch_count=x_train.shape[0]//100
        for i in range(batch_count):
            batch_xs, batch_ys= x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
            cost_val,_=sess.run([cost,optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            if i%20==0:
                print(i, cost_val)

    pred, corr, acc = sess.run([prediction, correction, accuracy], feed_dict={X:x_test, Y:y_test})
    print('pred : {}\t corr: {}\t acc : {}'.format(pred, corr, acc))