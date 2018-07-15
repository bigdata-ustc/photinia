"""
@Author: zhkun
@Time: 2018/07/13 20:53
@File: vgg16_new
@Description: 
@Something to attention: 
"""
import tensorflow as tf

import photinia as ph

import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16(ph.Widget):

    def __init__(self, name='vgg16'):
        self._height = 224
        self._width = 224
        super(Vgg16, self).__init__(name)

    def _build(self):
        # conv1 padding=SAME
        self._conv1_1 = ph.Conv2D(
            'conv1_1',
            input_size=[self._height, self._width, 3],
            output_channels=64,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        # conv1_2 padding=SAME
        self._conv1_2 = ph.Conv2D(
            'conv1_2',
            input_size=self._conv1_1.output_size,
            output_channels=64,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._pool1 = ph.Pool2D(
            'pool1',
            input_size=self._conv1_2.output_size,
            filter_height=2, filter_width=2, stride_height=2, stride_width=2,
            padding='SAME',
            pool_type='max'
        )
        #
        # conv2 padding=SAME
        self._conv2_1 = ph.Conv2D(
            'conv2_1',
            input_size=self._pool1.output_size,
            output_channels=128,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._conv2_2 = ph.Conv2D(
            'conv2_2',
            input_size=self._conv2_1.output_size,
            output_channels=128,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._pool2 = ph.Pool2D(
            'pool2',
            input_size=self._conv2_2.output_size,
            filter_height=2, filter_width=2, stride_height=2, stride_width=2,
            padding='SAME',
            pool_type='max'
        )
        #
        # conv3 padding=SAME
        self._conv3_1 = ph.Conv2D(
            'conv3_1',
            input_size=self._pool2.output_size,
            output_channels=256,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._conv3_2 = ph.Conv2D(
            'conv3_2',
            input_size=self._conv3_1.output_size,
            output_channels=256,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._conv3_3 = ph.Conv2D(
            'conv3_3',
            input_size=self._conv3_2.output_size,
            output_channels=256,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._pool3 = ph.Pool2D(
            'pool3',
            input_size=self._conv3_3.output_size,
            filter_height=2, filter_width=2, stride_height=2, stride_width=2,
            padding='SAME',
            pool_type='max'
        )
        #
        # conv4 padding=SAME
        self._conv4_1 = ph.Conv2D(
            'conv4_1',
            input_size=self._pool3.output_size,
            output_channels=512,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._conv4_2 = ph.Conv2D(
            'conv4_2',
            input_size=self._conv4_1.output_size,
            output_channels=512,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._conv4_3 = ph.Conv2D(
            'conv4_3',
            input_size=self._conv4_2.output_size,
            output_channels=512,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._pool4 = ph.Pool2D(
            'pool4',
            input_size=self._conv4_3.output_size,
            filter_height=2, filter_width=2, stride_height=2, stride_width=2,
            padding='SAME',
            pool_type='max'
        )
        #
        # conv5 padding=SAME
        self._conv5_1 = ph.Conv2D(
            'conv5_1',
            input_size=self._pool4.output_size,
            output_channels=512,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._conv5_2 = ph.Conv2D(
            'conv5_2',
            input_size=self._conv5_1.output_size,
            output_channels=512,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._conv5_3 = ph.Conv2D(
            'conv5_3',
            input_size=self._conv5_2.output_size,
            output_channels=512,
            filter_height=3, filter_width=3, stride_width=1, stride_height=1,
            padding='SAME'
        )
        self._pool5 = ph.Pool2D(
            'pool5',
            input_size=self._conv5_3.output_size,
            filter_height=2, filter_width=2, stride_height=2, stride_width=2,
            padding='SAME',
            pool_type='max'
        )
        #
        # fc layer
        self._fc6 = ph.Linear('fc6', input_size=self._pool5.flat_size, output_size=4096)
        self._fc7 = ph.Linear('fc7', input_size=self._fc6.output_size, output_size=4096)
        self._fc8 = ph.Linear(
            'fc8',
            input_size=self._fc7.output_size, output_size=1000,
            w_init=ph.init.RandomNormal(stddev=1e-4)
        )

    def _setup(self, x):
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        h = ph.setup(
            bgr,
            [self._conv1_1, tf.nn.relu, self._conv1_2, tf.nn.relu, self._pool1,
             self._conv2_1, tf.nn.relu, self._conv2_2, tf.nn.relu, self._pool2,
             self._conv3_1, tf.nn.relu, self._conv3_2, tf.nn.relu, self._conv3_3, tf.nn.relu, self._pool3,
             self._conv4_1, tf.nn.relu, self._conv4_2, tf.nn.relu, self._conv4_3, tf.nn.relu, self._pool4,
             self._conv5_1, tf.nn.relu, self._conv5_2, tf.nn.relu, self._conv5_3, tf.nn.relu, self._pool5,
             ph.ops.flatten,
             self._fc6, tf.nn.relu,
             self._fc7, tf.nn.relu]
        )
        y = self._fc8.setup(h)
        y = tf.nn.softmax(y)
        return y, h

    @staticmethod
    def _lrn(x):
        return tf.nn.local_response_normalization(
            x,
            depth_radius=1,
            alpha=1e-5,
            beta=0.75,
            bias=1.0
        )

    def load_pretrain(self, model_file='vgg16'):
        ph.io.load_model_from_tree(self, model_file)

    def load_parameters(self, model_file='vgg16.npy'):
        self.data_dict = np.load(model_file, encoding='latin1').item()
        param_dict = {}
        prepath = 'vgg16'

        for op_name in self.data_dict:
            for param in self.data_dict[op_name]:
                if len(param.shape) == 1:
                    param_dict[prepath + '/' + op_name + '/b:0'] = param
                else:
                    param_dict[prepath + '/' + op_name + '/w:0'] = param

        return param_dict


if __name__ == '__main__':
    dumper = ph.io.dumpers.TreeDumper('checkpoint')
    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True

    with tf.Session(config=configs) as sess:
        model = Vgg16()

        sess.run(tf.global_variables_initializer())

        pre_train_parameter = model.load_parameters()
        model.set_parameters(pre_train_parameter)

        # model.load_pretrain('/home/zhkun/pre_train_model/vgg16')



