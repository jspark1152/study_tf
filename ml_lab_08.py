import tensorflow as tf
import numpy as np

x=[
    [0,1,2],
    [2,1,0]
]

#axis=0 인 경우 가장 바깥쪽 축에 해당하므로 현재 세로를 나타냄
print(tf.argmax(x, axis=0).eval)

#axis=1 인 경우 Rank가 2이므로 가장 안쪽 축에 해당하므로 현재 가로를 나타냄
print(tf.argmax(x, axis=1).eval)

print(tf.argmax(x, axis=-1))

#t의 Rank=3 따라서 shape=(a, b, c)로 나타남
t=np.array([
    [
        [0,1,2],
        [3,4,5]
    ],
    [
        [6,7,8],
        [9,10,11]
    ]
])

print(t.shape)

#reshape을 통해 Rank=2로 다듬기가 가능하며 이때 shape은 (4,3)
print(tf.reshape(t, [-1,3]))

#reshape으로 원래 상태로 다시 복원도 가능
print(tf.reshape(t, [-1,2,3]))

#squeeze를 사용하면 Rank=1인 Array로 출력
print(tf.squeeze([[0],[1],[2]]))

#expand_dims을 사용하면 Rank값을 변경
print(tf.expand_dims([0,1,2], 1))

#one_hot은 one_hot 코딩. 단, 0 값을 [1, 0, ...] 의 형태로 변환하기 때문에 Rank 값이 1 증가함
print(tf.one_hot([[0],[3]], depth=4))

#cast는 해당 배열을 지정한 데이터 타입으로 출력. 즉, 데이터 타입을 반드시 입력.
print(tf.cast([1.9, 2.3, 3.5, 4.4, True], tf.int32))

a=[1,4]
b=[2,5]
c=[3,6]

#stack은 배열 순서대로 쌓음. 이때 누적 방향축 설정도 가능
print(tf.stack([a,b,c], axis=0))
print(tf.stack([a,b,c], axis=-1))

p=[[0,1,2],[2,1,0]]

#ones_like는 해당 텐서와 같은 쉐잎에 모든 성분값이 1, zeros_like는 모든 성분값이 0인 텐서를 출력
print(tf.ones_like(p))
print(tf.zeros_like(p))

#zip을 통해 각 배열의 성분 하나씩 출력 가능
for m, n in zip([1,2,3], [4,5,6]):
    print(m, n)