import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data=[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data=[[0],[0],[0],[1],[1],[1]]

X=tf.placeholder(tf.float32,shape=[None,2])
Y=tf.placeholder(tf.float32,shape=[None,1])
W=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.sigmoid(tf.matmul(X,W)+b)

cost=tf.reduce_mean(-Y*tf.log(hypothesis)+(Y-1)*tf.log(1-hypothesis))

Prediction=tf.cast(hypothesis>0.5,dtype=tf.float32)
Accuracy=tf.reduce_mean(tf.cast(tf.equal(Prediction,Y),dtype=tf.float32))

train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, _=sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
    if step%200==0:
        print(step, cost_val)

h,c,a=sess.run([hypothesis, Prediction, Accuracy],feed_dict={X:x_data,Y:y_data})

print("\nHypothesis: ",h,"\nCorrect(Y): ",c,"Accuracy: ",a)