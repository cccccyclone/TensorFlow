from tensorflow.examples.tutorials.mnist import input_data
#导入mnist数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#输出训练集、测试集、验证集的大小和维度信息
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)