import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np

xy=np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

#training_set
x_data=xy[0:10,0:-1]
y_data=xy[0:10,[-1]]
#test_set
x_test=xy[10:,0:-1]
y_test=xy[10:,[-1]]

print(y_data)

nb_classes=7

X=tf.placeholder(tf.float32, shape=[None,16])
Y=tf.placeholder(tf.int32, shape=[None,1])

Y_one_hot=tf.one_hot(Y, nb_classes)
Y_one_hot=tf.reshape(Y_one_hot, [-1, nb_classes])

W=tf.Variable(tf.random_normal([16,nb_classes]), name='weight')
b=tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits=tf.matmul(X,W)+b
hypothesis=tf.nn.softmax(logits)

cost=tf.reduce_mean(tf.reduce_sum(-Y_one_hot*tf.log(hypothesis), axis=1))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction=tf.argmax(hypothesis, 1)
correction=tf.cast(tf.equal(prediction, tf.argmax(Y_one_hot, 1)), tf.float32)
accuracy=tf.reduce_mean(correction)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20000):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step%1000==0:
            loss, hyp, pred, acc = sess.run([cost, hypothesis, prediction, accuracy], feed_dict={X:x_data, Y:y_data})
            print('Step : {:5}\t Loss : {:.3f}\t Accuracy : {:.3f}'.format(step, loss, acc))
    #결과물을 보면 training_set이 적절치 않은걸 확인할 수 있음
    pred, corr, acc = sess.run([prediction, correction, accuracy], feed_dict={X:x_test, Y:y_test})
    print('\nTest Result\n Prediction : {}\n Correction : {}\t Accuracy : {}'.format(pred, corr, acc))