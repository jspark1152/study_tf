import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#불러올 파일의 양이 방대할경우 큐러너를 사용
#파일들의 이름과 셔플의 여부를 정의
filename_queue=tf.train.string_input_producer(['data-03-diabetes.csv'],shuffle=False,name='filename_queue')

#리더로 어떤 파일을 읽을지 정의하고 key,value 값을 통해 데이터를 읽어들임
reader=tf.TextLineReader()
key, value=reader.read(filename_queue)

#읽어들일 행의 디폴트값을 정의/현재 읽어들일 파일의 데이터는 각 행이 9열로 이루어져있기 때문에 9개의 플로팅 값으로 이중 리스트 형식을 취함
record_defaults=[[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
#리더에 들어있는 값들을 디코드해주어 xy에 입력 > xy는 결과적으로 각 열들이 들어있는 이중 리스트 구조. 각 리스트는 열벡터와 동일
xy=tf.decode_csv(value, record_defaults=record_defaults)
#슬라이스하는 과정
train_x_batch,train_y_batch=tf.train.batch([xy[0:-1],xy[-1:]], batch_size=10)

X=tf.placeholder(tf.float32, shape=[None,8])
Y=tf.placeholder(tf.float32, shape=[None,1])
W=tf.Variable(tf.random_normal([8,1]), name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.sigmoid(tf.matmul(X,W)+b)

cost=tf.reduce_mean(-Y*tf.log(hypothesis)-(1-Y)*tf.log(1-hypothesis))

predicted=tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

#큐러너를 사용할 때 시작 전 단계의 코드.
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(10001):
    x_batch,y_batch=sess.run([train_x_batch,train_y_batch])
    cost_val, _=sess.run([cost, train], feed_dict={X:x_batch,Y:y_batch})
    if step%200==0:
        print(step, cost_val)

#큐러너를 종료하는 구문
coord.request_stop()
coord.join(threads)

h,p,a=sess.run([hypothesis,predicted,accuracy], feed_dict={X:x_batch,Y:y_batch})
print("\nHypothesis :\n",h,"\nCorrect(Y) :\n",p,"\nAccuracy : ",a)