import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np

xy=np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

#range(Y)={0,1,2,3,4,5,6}
np_classes=7

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X=tf.placeholder(tf.float32, shape=[None,16])
Y=tf.placeholder(tf.int32, shape=[None,1])
W=tf.Variable(tf.random_normal([16, np_classes]), name='weight')
b=tf.Variable(tf.random_normal([1, np_classes]), name='bias')

Y_one_hot=tf.one_hot(Y, np_classes)
Y_one_hot=tf.reshape(Y_one_hot, [-1, np_classes])

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

cost=tf.reduce_mean(tf.reduce_sum(-Y_one_hot*tf.log(hypothesis), axis=1))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

prediction=tf.argmax(hypothesis, 1)
correction=tf.cast(tf.equal(prediction, tf.argmax(Y_one_hot, 1)), tf.float32)
accuracy=tf.reduce_mean(correction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        cost_val, _=sess.run([cost, optimizer], feed_dict={X:x_data, Y:y_data})
        if step%50==0:
            print(step, cost_val)

    pred, corr, acc=sess.run([prediction, correction, accuracy], feed_dict={X:x_data, Y:y_data})
    print('Predcition : {}\nCorrection : {}\nAccuracy : {}'.format(pred,corr,acc))