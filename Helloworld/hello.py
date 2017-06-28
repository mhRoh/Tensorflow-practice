import tensorflow as tf

sess = tf.Session()
hello = tf.constant('Hello, Tensorflow')


print(sess.run(hello))
print(str(sess.run(hello), encoding='utf-8'))




