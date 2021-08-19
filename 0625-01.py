import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

W=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))
#placeholder를 이용하여 X,Y 값을 직접 입력
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#hypothesis 설정
hypothesis=W*X+b
#실제 값과의 차이를 위해 cost 설정
cost=tf.reduce_mean(tf.square(hypothesis-Y))
#GradientDescent Algorithm을 이용
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
#학습의 최종 목표는 cost의 최소화
train=optimizer.minimize(cost)

sess=tf.Session()
#변수들 사용전 초기화 필수
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _=sess.run([cost, W, b, train],
        feed_dict={X:[1,2,3,4,5],
            Y:[2.1,3.1,4.1,5.1,6.1]})
    if step %20==0:
        print(step, cost_val, W_val, b_val)
