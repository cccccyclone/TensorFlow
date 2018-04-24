######################
#感知机MNIST数据集测试
#不同于tensor,Variable的值会一直保留下来
######################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#导入mnist数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#注册为默认的session，之后的运算也默认跑在这个session里，不同session的数据和运算应该是独立的
sess = tf.InteractiveSession()
#定义输入数据，第二个参数为数据尺寸，None表示不限条数的输入，784代表每条输入是一个784维向量
x = tf.placeholder(tf.float32,[None,784])
#不同于tensor,Variable的值会一直保留下来
W = tf.Variable(tf.zeros([784,10]))
#因为最后是分成十类
b = tf.Variable(tf.zeros([10]))
#网络计算得出的值
y = tf.nn.softmax(tf.matmul(x,W)+ b )
#真实标签，[0...1..0]的形式
y_ = tf.placeholder(tf.float32,[None,10])
#定义损失函数,reduction_indices意思是将矩阵的维度进行压缩
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#定义学习率和损失函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#全局参数初始化器
tf.global_variables_initializer().run()
#使用mini_batch对样本随机梯度下降
for i in range(1000):
    #取出数据，100项为1组
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

    #完成训练，开始验证,argmax()表示返回最大值下标，1代表的是行内比较，0代表列内比较
    correct_prediction = tf.equal(tf.argmax(y,1),tf.arg_max(y_,1))
    #统计全部样本预测的arruracy,先用tf.cast将之前correct_prediction输出的bool值转换为float32
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #eval()函数用于启动计算，用测试集数据做检测
    print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
