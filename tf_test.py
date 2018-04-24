import tensorflow as tf

##????未解决
'''
a = tf.constant([[
    [   [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [8.0, 7.0, 6.0, 5.0],
        [4.0, 3.0, 2.0, 1.0]    ],
    [   [9.0, 3.0, 10.0, 1.0],
        [8.0, 7.0, 6.0, 5.0],
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]    ]
]])
#为什么非要reshape
#a = tf.reshape(a, [1, 4, 4, 2])

pooling = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    print("image:")
    image = sess.run(a)
    print(image)
    print("reslut:")
    result = sess.run(pooling)
    print(result)
'''
