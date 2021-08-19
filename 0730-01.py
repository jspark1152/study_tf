import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X=tf.placeholder(tf.float32, shape=[None,2])
Y=tf.placeholder(tf.float32, shape=[None,1])
W=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

#hypothesis using sigmoid:tf.div(1.,1.+tf.exp(tf.matmul(X,W)+b))
hypothesis=tf.sigmoid(tf.matmul(X,W)+b)