import tensorflow as tf

'''
우선 이 소스에서는 feeding이 어떻게 이루어 지는 과정을 좀 살펴 볼 필요가 있다.
TensorFlow는 Graph를 이용하여 이를 처리하고 있다고 하는데 이 의미는 무엇인가?
'''

tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

'''
위에서 X,Y에 대해 각각 float32로 placeholder를 선언 하였으니 이는 값을 가지고 있다가 Operation으로 보내는
Node라고 보면 되는 건가?
Here, we declare X, Y as a placeholder to hold a float32 value. So can we consider it as a node?
'''
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1, 2, 3], Y : [1, 2, 3]})

    if step == 2000 :
        print(step, cost_val, W_val, b_val)

print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 1.1]}))

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X : [1, 2, 3, 4, 5], Y : [2.1, 3.1, 4.1, 5.2, 6.3]})

    if step == 2000 :
        print(step, cost_val, W_val, b_val)

print(sess.run(hypothesis, feed_dict={X: [5]})) #Upon the result of training, our hypotesis inform us the answer of '5' is....what...
print(sess.run(hypothesis, feed_dict={X: [1.5, 1.1]}))
