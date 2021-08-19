import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

xy=np.loadtxt('data-03-diabetes.csv',delimiter=',',dtype=np.float32)

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X=tf.placeholder(tf.float32, shape=[None,8])
Y=tf.placeholder(tf.float32, shape=[None,1])
W=tf.Variable(tf.random_normal([8,1]), name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.sigmoid(tf.matmul(X,W)+b)

cost=tf.reduce_mean(-Y*tf.log(hypothesis)-(1-Y)*tf.log(1-hypothesis))

prediction=tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(prediction,Y), dtype=tf.float32))

train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, _=sess.run([cost, train], feed_dict={X:x_data,Y:y_data})
    if step%200==0:
        print(step, cost_val)

h,p,a=sess.run([hypothesis,prediction,accuracy], feed_dict={X:x_data,Y:y_data})
print("\nHypothesis :\n",h,"\nCorrect(Y) :\n",p,"\nAccuracy : ",a)