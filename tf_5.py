######################
#卷积神经网络初体验
#74-76不错的介绍
######################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
#导入mnist数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#注册为默认的session，之后的运算也默认跑在这个session里，不同session的数据和运算应该是独立的
sess = tf.InteractiveSession()

#创建权值初始化函数
def weight_variable(shape):
    #截断正态分布，打破完全对称
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#创建偏置初始化函数
def bias_variable(shape):
    #因为用Relu激活函数，要避免死亡节点
    #标准差为0.1，大小为shape
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#对于输入x，应该为[height,width,channel,num_of_kernel],padding=SAME是补全输入图像，使得卷积核中心能够到达输入的边界
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#一般要求strides[0]=strides[3]=1
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
#-1表示样本数量固定不变，1表示1通道
x_image = tf.reshape(x,[-1,28,28,1])
# 1通道，卷积核数量为32
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#定义第二个卷积层，通道数变为32，因为通道数和卷积核的数量相同
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#由于使用SAME的padding方式，经过两次卷积操作图像大小没变，而两次池化使得图像变成7*7大小
#现在tensor尺寸为7*7*64，转化为1维向量，进行全连接操作
#最后隐含层的权重初始化
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
#向量展开
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#为减轻过拟合，使用 随机失活
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#最后连接softmax层，得到最后的概率输出
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#定义损失函数,reduction_indices意思是将矩阵的维度进行压缩
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
#自适应学习率调整，使用比较小的学习速率
train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(y_, 1))
# 统计全部样本预测的arruracy,先用tf.cast将之前correct_prediction输出的bool值转换为float32
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 ==0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d, training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
