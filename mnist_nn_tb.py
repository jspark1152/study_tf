import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np

mnist=tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train=x_train/255
x_train=np.reshape(x_train, [-1, 784])
x_train=np.array(x_train, np.float32)

x_test=x_test/255
x_test=np.reshape(x_test, [-1, 784])
x_test=np.array(x_test, np.float32)

y_train=np.reshape(y_train, [-1, 1])
y_train=np.array(y_train, np.int32)

y_test=np.reshape(y_test, [-1, 1])
y_test=np.array(y_test, np.int32)

print(x_train.dtype, y_train.dtype, x_test.shape, y_test.shape)

nb_classes=10

#Tensorboard
#1.무엇을 보고 싶은지 > tf.summary.scalar(name,scalr) / tf.summary.image(name,image) / tf.summary.histogram(name,histogram)
#2.어디에 기록할건지 > tf.summary.merge_all() / tf.summary.merge(summaries) / tf.summary.FileWriter(log_dir,graph)
#3.언제마다 기록할건지 > summary=sess.run(merge) / writer.add_symmary(summary,global_step)
#4.Tensorboard 오픈

X=tf.placeholder(tf.float32, shape=[None, 784])
Y=tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot=tf.one_hot(Y, nb_classes)
Y_one_hot=tf.reshape(Y_one_hot, [-1, nb_classes])

#Name scope가 넓을수록 시각화에 유리
with tf.name_scope('layer') as scope:
    W=tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
    b=tf.Variable(tf.random_normal([1, nb_classes]), name='bias')
    
    W_hist=tf.summary.histogram('weights', W)
    b_hist=tf.summary.histogram('biases', b)
    
    hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)
    hypothesis_hist=tf.summary.histogram('hypothesis', hypothesis)

with tf.name_scope('cost') as scope:
    cost=tf.reduce_mean(tf.reduce_sum(-Y_one_hot*tf.log(hypothesis), axis=1))
    cost_summ=tf.summary.scalar('cost', cost)

with tf.name_scope('optimizer') as scope:
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction=tf.argmax(hypothesis,1)
correction=tf.cast(tf.equal(prediction, tf.argmax(Y_one_hot, 1)), tf.float32)
accuracy=tf.reduce_mean(correction)

accuracy_summ=tf.summary.scalar('accuracy', accuracy)

summary=tf.summary.merge_all()

batch_size=100
epochs=15

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer=tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)
    
    for epoch in range(epochs):
        batch_count=x_train.shape[0]//100
        for i in range(batch_count):
            batch_xs, batch_ys= x_train[i*batch_size:(i+1)*batch_size,:], y_train[i*batch_size:(i+1)*batch_size,:]
            cost_val,s,_=sess.run([cost,summary,optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            
            writer.add_summary(s, global_step=i)
            
            
            if i%100==0:
                print(i, cost_val)
        
    pred, corr, acc = sess.run([prediction, correction, accuracy], feed_dict={X:x_test, Y:y_test})
    print('pred : {}\t corr: {}\t acc : {}'.format(pred, corr, acc))