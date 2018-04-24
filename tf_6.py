######################
#卷积神经网络进阶体验
######################

#对weights进行了L2正则化
#对图片翻转、随机剪切等数据增强，制造了更多样本
#在每个卷积-最大池化层后面使用了LRN层，增强泛化能力

import  cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
#下载数据的默认路径
data_dir = '/temp/cifar10_data/cifar-10-batches-bin'

#使用wl控制L2 loss的大小
def variable_with_loss(shape,stddev,wl):
    #截断正态分布
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        #保存进一个字典
        tf.add_to_collection('losses',weight_loss)
    return var

#数据下载
#cifar10.maybe_download_and_extract()

#获得数据增强后的训练数据
images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
#获得测试数据，不需增强，但需要裁剪图片正中间的24x24大小的区块，并进行数据标准化操作
images_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

##定义输入数据
image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder = tf.placeholder(tf.int32,[batch_size])

##定义第一层
# wl=0.0表示不使用正则化
weight1 = variable_with_loss(shape=[5,5,3,64],stddev=5e-2,wl=0.0)
#卷积操作
kernel1 = tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME')
#偏置初始化为0
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
#加入LRN层
norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

##定义第二层
# wl=0.0表示不使用正则化
weight2= variable_with_loss(shape=[5,5,64,64],stddev=5e-2,wl=0.0)
kernel2 = tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME')
#偏置初始化为0.1
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
#调换了LRN层和池化层的顺序
norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
##全连接层
##这里的reshape为什么用batch_size????
reshape = tf.reshape(pool2,[batch_size,-1])
#计算扁平化之后的维度
dim = reshape.get_shape()[1].value
weight3 = variable_with_loss(shape=[dim,384],stddev=0.04,wl = 0.004)
bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bias3)

##全连接层2,节点数目减少一半
weight4 = variable_with_loss(shape=[384,192],stddev=0.04,wl=0.004)
bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3,weight4)+bias4)

##最后一层，这里不用softmax，直接比较大小看类别
weight5 = variable_with_loss(shape=[192,10],stddev=1/192.0,wl=0.0)
bias5 = tf.Variable(tf.constant(0.0,shape=[10]))
logits = tf.add(tf.matmul(local4,weight5),bias5)

#计算损失时用到softmax
def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses',name='total_loss'))

loss = loss(logits,label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_top = tf.nn.in_top_k(logits,label_holder,1)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run([images_train,labels_train])
    _,loss_value = sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
    duration = time.time() -start_time
    if step%10 ==0:
        examples_per_sec = batch_size/duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)')
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))

num_examples = 10000
import math
num_iter = int(math.ceil(num_examples/batch_size))
true_count = 0
total_sample_count = num_iter*batch_size
step = 0
while step < num_iter:
    image_batch,label_batch = sess.run([images_test,images_test])
    predictions = sess.run([top_k_top],feed_dict={image_holder:image_batch,label_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count/total_sample_count
print('precision @ 1 = %.3f' % precision)