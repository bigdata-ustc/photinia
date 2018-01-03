# import tensorflow as tf
#
# import numpy as np
#
#
# def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):  # activation_function=None线性函数
#     layer_name = "layer%s" % n_layer
#     with tf.name_scope(layer_name):
#         with tf.name_scope('weights'):
#             Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # Weight中都是随机变量
#             tf.summary.histogram(layer_name + "/weights", Weights)  # 可视化观看变量
#         with tf.name_scope('biases'):
#             biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # biases推荐初始值不为0
#             tf.summary.histogram(layer_name + "/biases", biases)  # 可视化观看变量
#         with tf.name_scope('Wx_plus_b'):
#             Wx_plus_b = tf.matmul(inputs, Weights) + biases  # inputs*Weight+biases
#             tf.summary.histogram(layer_name + "/Wx_plus_b", Wx_plus_b)  # 可视化观看变量
#         if activation_function is None:
#             outputs = Wx_plus_b
#         else:
#             outputs = activation_function(Wx_plus_b)
#         tf.summary.histogram(layer_name + "/outputs", outputs)  # 可视化观看变量
#         return outputs
#
#
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # [-1,1]区间，300个单位，np.newaxis增加维度
# noise = np.random.normal(0, 0.05, x_data.shape)  # 噪点
# y_data = np.square(x_data) - 0.5 + noise
# with tf.name_scope('inputs'):  # 结构化
#     xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
#     ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
# # 三层神经，输入层（1个神经元），隐藏层（10神经元），输出层（1个神经元）
# l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)  # 隐藏层
# prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)  # 输出层
# # predition值与y_data差别
# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(
#         tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # square()平方,sum()求和,mean()平均值
#     loss_sum = tf.summary.scalar('loss', loss)  # 可视化观看常量
#
# x = tf.placeholder(shape=(), dtype=tf.float32)
# lll = loss * x
# tf.summary.scalar('lll', lll)
#
# with tf.name_scope('train'):
#     train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 0.1学习效率,minimize(loss)减小loss误差
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# # 合并到Summary中
# merged = tf.summary.merge([loss_sum])
# # 选定可视化存储目录
#
# writer = tf.summary.FileWriter("/dev/shm/dd", sess.graph)
#
# sess.run(init)  # 先执行init
#
# # 训练1k次
#
# for i in range(10000):
#     print(i)
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#
#     if i % 50 == 0:
#         result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})  # merged也是需要run的
#
#         writer.add_summary(result, i)  # result是summary类型的，需要放入writer中，i步数（x轴）


import numpy as np
import tensorflow as tf

"""
Maxout OP from https://arxiv.org/abs/1302.4389
Max pooling is performed in given filter/channel dimension. This can also be
used after fully-connected layers to reduce number of features.
Args:
    inputs: A Tensor on which maxout will be performed
    num_units: Specifies how many features will remain after max pooling at the
      channel dimension. This must be multiple of number of channels.
    axis: The dimension where max pooling will be performed. Default is the
      last dimension.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
Returns:
    A `Tensor` representing the results of the pooling operation.
Raises:
    ValueError: if num_units is not multiple of number of features.
"""


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


import photinia as ph
from photinia import utils as phutils
from photinia import apps as phapps

if __name__ == '__main__':
    n = 100000000
    for i in range(n):
        phutils.print_progress(i + 1, n)
    exit()
    with tf.Session() as sess:
        x = tf.Variable(np.random.uniform(size=(25, 10, 500)))
        y = tf.square(x)
        mo = max_out(x, 5, axis=2)
        x = tf.Variable(
            [2, 2, 2, 2],
            dtype=tf.float32
        )
        k = tf.placeholder(tf.float32)
        y = tf.nn.dropout(x, k)
        z = tf.nn.dropout(x, k)
        sess.run(tf.global_variables_initializer())
        r = sess.run((y, z), feed_dict={k: 0.5})
        print(r)
