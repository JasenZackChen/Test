import tensorflow as tf

tup=tf.constant((1,23,3))
hello=tf.constant('hello world')
a=tf.constant((1,23,3))
b=tf.constant((2,34,4))
c=tf.add(a,b)
with tf.Session() as sess:
    print(sess.run(tup))
    print(sess.run(hello))
    print(sess.run(c))
