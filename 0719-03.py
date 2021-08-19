import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X=[1,2,3]
Y=[1,2,3]

W=tf.Variable(5.0)

hypothesis=X*W

gradient=tf.reduce_mean((W*X-Y)*X)*2

cost=tf.reduce_mean(tf.square(hypothesis-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs=optimizer.compute_gradients(cost)

apply_gradients=optimizer.apply_gradients(gvs)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

#optimizer 결과와 gradient를 직접 계산한 결과가 동일한 것을 확인
for step in range(10):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)