######################
#一个隐层的MLP测试
######################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
#导入mnist数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#注册为默认的session，之后的运算也默认跑在这个session里，不同session的数据和运算应该是独立的
sess = tf.InteractiveSession()

int_units = 784
h1_units = 300
#权重初始化为截断的正态分布
#加噪声来打破完全对称并避免0梯度，有时候还需给偏置赋上一些小的非零值来避免死亡神经元
W1 = tf.Variable(tf.truncated_normal([int_units,h1_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32,[None,int_units])
#训练和预测时的失活率是不同的
keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
#设置随机失活
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
#获取结果类别
y = tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
#真实标签
y_ = tf.placeholder(tf.float32,[None,10])
#定义损失函数,reduction_indices意思是将矩阵的维度进行压缩
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#自适应学习率调整
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#全局参数初始化器
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
    #完成训练，开始验证,argmax()表示返回最大值下标，1代表的是行内比较，0代表列内比较
    correct_prediction = tf.equal(tf.argmax(y,1),tf.arg_max(y_,1))
    #统计全部样本预测的arruracy,先用tf.cast将之前correct_prediction输出的bool值转换为float32
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #eval()函数用于启动计算，用测试集数据做检测
    print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))


