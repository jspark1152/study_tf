import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

xy=np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X=tf.placeholder(tf.float32, shape=[None,3])
Y=tf.placeholder(tf.float32, shape=[None,1])
W=tf.Variable(tf.random_normal([3,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

hypothesis=tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        cost_val, _= sess.run([cost, optimizer], feed_dict={X:x_data, Y:y_data})

        if step%50==0:
            print('{} \t {}'.format(step, cost_val))