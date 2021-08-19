import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np

xy=np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

#Y의 데이터 값의 범위가 0,1,...,6
nb_classes=7

X=tf.placeholder(tf.float32, [None,16])
Y=tf.placeholder(tf.int32, [None,1])

#One-Hot encoding을 위한 작업이 필요
#tf.one_hot 함수를 이용 > 현재 Y 데이터의 범위는 0~6 이므로 [3] 의 값은 [[0 0 0 1 0 0 0]] 으로 변환
Y_one_hot=tf.one_hot(Y, nb_classes)
#tf.reshape 함수를 이용하여 위의 결과로 2X7의 값이 생기므로 다시 1X7로의 변환이 필요 
Y_one_hot=tf.reshape(Y_one_hot, [-1, nb_classes])

#X=1X16 Y_one_hot=1X7(=nb_classes) 따라서 W는 16X7(=nb_classes)
W=tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b=tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits=tf.matmul(X,W)+b
hypothesis=tf.nn.softmax(logits)

#TF에서 제공되는 cross entropy 함수 형태 = tf.reduce_sum(-Y*tf.log(hypothesis)))
#cost_i=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost_i=tf.reduce_sum(-Y_one_hot*tf.log(hypothesis), axis=1)

cost=tf.reduce_mean(cost_i)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#예측값을 argmax 함수를 통해 계산
prediction=tf.argmax(hypothesis,1)
correction=tf.cast(tf.equal(prediction, tf.argmax(Y_one_hot, 1)), tf.float32)
accuracy=tf.reduce_mean(correction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step%100==0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            print('Step : {:5}\tLoss : {:.3f}\tAcc : {:.2f}'.format(step, loss, acc))

    pred=sess.run(prediction, feed_dict={X:x_data})
    #flatten 함수로 열벡터를 행벡터 표현으로 변환 가능
    for p, y in zip(pred, y_data.flatten()):
        print('[{}] Prediction: {} True Y: {}'.format(p == int(y), p, int(y)))
