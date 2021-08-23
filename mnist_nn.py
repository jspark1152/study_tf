import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np
from tensorflow import keras

#라벨 : 0,1,2,...,9 까지 총 10개
nb_classes=10

#구글링하면 문제 해결 다 가능한듯..
#mac에서 mnist데이터 로드 과정에서 이슈 발생 > 파이썬 certification 업데이트로 해결
#mnist 데이터를 불러들이는 과정
mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)=mnist.load_data()

#각 자료의 shape을 확인 > Train Set의 경우 60000개의 데이터가 있으며 각 데이터는 28X28 형태, Test Set의 경우 10000개의 데이터가 있음
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#x의 데이터 값들이 RGB 값으로 0~255를 가지는데 이를 0과 1사이의 값으로 스케일링
#또한 현재 28X28 형태의 행렬 배열을 reshape을 통해 하나의 배열로 변환
#현재 문제가 데이터 타입이 tf.reshape을 사용하면 float64이기 때문에 이를 처리해야함
x_train=x_train/255
x_train=np.reshape(x_train, [-1, 784])
x_train=np.array(x_train, np.float32)

y_train=np.reshape(y_train, [-1, 1])
y_train=np.array(y_train, np.int32)
print(x_train, y_train)

x_test=x_test/255
x_test=np.reshape(x_test, [-1, 784])
x_test=np.array(x_test, np.float32)

y_test=np.reshape(y_test, [-1, 1])
y_test=np.array(y_test, np.int32)

X=tf.placeholder(tf.float32, shape=[None, 784])
Y=tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot=tf.one_hot(Y, nb_classes)
Y_one_hot=tf.reshape(Y_one_hot, [-1, nb_classes])

#W=tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
#b=tf.Variable(tf.random_normal([1, nb_classes]), name='bias')

#Neural Network
W1=tf.Variable(tf.random_normal([784,100]), name='weight1')
b1=tf.Variable(tf.random_normal([1,100]), name='bias1')
layer1=tf.nn.softmax(tf.matmul(X,W1)+b1)

W2=tf.Variable(tf.random_normal([100,80]), name='weight2')
b2=tf.Variable(tf.random_normal([1,80]), name='bias2')
layer2=tf.nn.softmax(tf.matmul(layer1,W2)+b2)

W3=tf.Variable(tf.random_normal([80,60]), name='weight3')
b3=tf.Variable(tf.random_normal([1,60]), name='bias3')
layer3=tf.nn.softmax(tf.matmul(layer2,W3)+b3)

W4=tf.Variable(tf.random_normal([60,40]), name='weight4')
b4=tf.Variable(tf.random_normal([1,40]), name='bias4')
layer4=tf.nn.softmax(tf.matmul(layer3,W4)+b4)

W5=tf.Variable(tf.random_normal([40,20]), name='weight5')
b5=tf.Variable(tf.random_normal([1,20]), name='bias5')
layer5=tf.nn.softmax(tf.matmul(layer4,W5)+b5)

W6=tf.Variable(tf.random_normal([20,nb_classes]), name='weight6')
b6=tf.Variable(tf.random_normal([1,nb_classes]), name='bias6')

#softmax를 통해 텐서의 값들의 합이 1이 되게끔 변환
#hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

hypothesis=tf.nn.softmax(tf.matmul(layer5,W6)+b6)


cost=tf.reduce_mean(tf.reduce_sum(-Y_one_hot*tf.log(hypothesis), axis=1))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#원핫코딩을 통해 예측값 조사
prediction=tf.argmax(hypothesis,1)
correction=tf.cast(tf.equal(prediction, tf.argmax(Y_one_hot, 1)), tf.float32)
accuracy=tf.reduce_mean(correction)


batch_size=100
epochs=200

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        batch_count=x_train.shape[0]//100
        for i in range(batch_count):
            batch_xs, batch_ys= x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
            cost_val,_=sess.run([cost,optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            if i%100==0:
                print(i, cost_val)

    pred, corr, acc = sess.run([prediction, correction, accuracy], feed_dict={X:x_test, Y:y_test})
    print('pred : {}\t corr: {}\t acc : {}'.format(pred, corr, acc))