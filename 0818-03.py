import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np

xy=np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

classes=7

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X=tf.placeholder(tf.float32, shape=[None,16])
Y=tf.placeholder(tf.int32, shape=[None,1])

Y_one_hot=tf.one_hot(Y, classes)
Y_one_hot=tf.reshape(Y_one_hot, [-1,classes])

W=tf.Variable(tf.random_normal([16, classes]), name='weight')
b=tf.Variable(tf.random_normal([classes]), name='weight')

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

cost=tf.reduce_mean(tf.reduce_sum(-Y_one_hot*tf.log(hypothesis), axis=1))


optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        cost_val, _=sess.run([cost, optimizer], feed_dict={X:x_data, Y:y_data})

        if step%50==0:
            print(step, cost_val)
    





