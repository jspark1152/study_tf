import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

x_data=[[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
#One-Hot Encoding : 하나의 1을 Hot point에 지정
y_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

X=tf.placeholder("float", [None,4])
Y=tf.placeholder("float", [None,3])
nb_classes=3

W=tf.Variable(tf.random_normal([4,nb_classes]), name='weight')
b=tf.Variable(tf.random_normal([nb_classes]), name='bias')

#Softmax 함수
hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

#cost 형태는 cross entropy
cost=tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypothesis), axis=1))

train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, _ = sess.run([cost,train], feed_dict={X:x_data, Y:y_data})
        if step %200 ==0:
            print(step, cost_val)

    all=sess.run(hypothesis, feed_dict={X:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
    #tf.arg_max 함수가 가장 큰 위치를 리턴.
    print(all, sess.run(tf.arg_max(all,1)))