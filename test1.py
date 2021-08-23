import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np

xy=np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X=tf.placeholder(tf.float32, shape=[None,3])
Y=tf.placeholder(tf.float32, shape=[None,1])
W=tf.Variable(tf.random_normal([3,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')
W_hist=tf.summary.histogram('weights',W)
b_hist=tf.summary.histogram('biases',b)

hypothesis=tf.matmul(X,W)+b
hypothesis_hist=tf.summary.histogram('hypothesis',hypothesis)

cost=tf.reduce_mean(tf.square(hypothesis-Y))
cost_summ=tf.summary.scalar('cost', cost)

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

summary=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)

    for step in range(2000):
        cost_val,s, _ = sess.run([cost,summary,optimizer], feed_dict={X:x_data, Y:y_data})
        writer.add_summary(s, global_step=step)
        if step%10 == 0:
            print(step, "Cost : ", cost_val)
    
    hyp = sess.run(hypothesis, feed_dict={X:[[80,70,105]]})
    print('\nPrediction of Final exam score : {}'.format(hyp))