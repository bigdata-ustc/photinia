#!/usr/bin/env python3

"""
@author: xi
@since: 2017-03-29
"""

import os
import shutil
import sys

import gflags
import tensorflow as tf

import photinia


class Encoder(photinia.Widget):
    """Encoder
    """

    def __init__(self,
                 name,
                 height,
                 width,
                 depth,
                 emb_size):
        self._height = height
        self._width = width
        self._depth = depth
        self._emb_size = emb_size
        super(Encoder, self).__init__(name)

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def depth(self):
        return self._depth

    @property
    def emb_size(self):
        return self._emb_size

    def _build(self):
        cnn = photinia.CNN(
            'cnn',
            self._height,
            self._width,
            self._depth,
            [(5, 5, 32, 2, 2),
             (5, 5, 64, 2, 2),
             (5, 5, 128, 2, 2),
             (5, 5, 256, 2, 2),
             (5, 5, 512, 2, 2)],
            with_batch_norm=True
        )
        self._cnn = cnn
        self._flat_size = cnn.flat_size
        self._emb = photinia.Linear('emb', self._flat_size, self._emb_size)

    def _setup(self, image):
        feature = self._cnn.setup(image)
        emb = self._emb.setup(feature)
        return emb


class Decoder(photinia.Widget):
    """Decoder
    """

    def __init__(self,
                 name,
                 height,
                 width,
                 depth,
                 emb_size):
        self._height = height
        self._width = width
        self._depth = depth
        self._emb_size = emb_size
        super(Decoder, self).__init__(name)

    def _build(self):
        height, width, depth = self._height, self._width, 512
        for _ in range(5):
            height, width = -(-height // 2), -(-width // 2)
        #
        flat_size = height * width * depth
        self._lin = photinia.Linear(
            'lin',
            self._emb_size,
            flat_size,
            with_batch_norm=True
        )
        self._tcnn = photinia.TransCNN(
            'tcnn',
            height,
            width,
            depth,
            [(5, 5, 256, 2, 2),
             (5, 5, 128, 2, 2),
             (5, 5, 64, 2, 2),
             (5, 5, 32, 2, 2),
             (5, 5, 3, 2, 2)],
            with_batch_norm=True
        )

    def _setup(self, emb):
        feature = self._lin.setup(emb)
        feature = photinia.lrelu(feature)
        image = self._tcnn.setup(feature)
        image = tf.nn.tanh(image)
        return image


class DCAE(photinia.Widget):
    """DCAE
    
    ./
        encoder/
        decoder/
    """

    def __init__(self,
                 name,
                 height,
                 width,
                 depth,
                 emb_size):
        self._height = height
        self._width = width
        self._depth = depth
        self._emb_size = emb_size
        super(DCAE, self).__init__(name)

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def depth(self):
        return self._depth

    @property
    def emb_size(self):
        return self._emb_size

    def _build(self):
        #
        # Encode.
        image = tf.placeholder(
            shape=(None, self._height, self._width, self._depth),
            dtype=photinia.D_TYPE
        )
        self._image = image
        encoder = Encoder(
            'encoder',
            self._height,
            self._width,
            self._depth,
            self._emb_size
        )
        self._encoder = encoder
        emb = encoder.setup(image)
        self._emb = emb
        #
        # Decode.
        decoder = Decoder(
            'decoder',
            self._height,
            self._width,
            self._depth,
            self._emb_size
        )
        self._decoder = decoder
        rec = decoder.setup(emb)
        self._rec = rec
        #
        # Loss.
        loss = tf.reduce_sum((image - rec) ** 2, (1, 2, 3))
        loss = tf.reduce_mean(loss)
        self._loss = loss

    @property
    def image(self):
        return self._image

    @property
    def encoder(self):
        return self._encoder

    @property
    def emb(self):
        return self._emb

    @property
    def decoder(self):
        return self._decoder

    @property
    def rec(self):
        return self._rec

    @property
    def loss(self):
        return self._loss

    def _setup(self):
        pass


class Model(photinia.Trainer):
    """DCAE
    """

    def __init__(self,
                 name,
                 session,
                 height,
                 width,
                 depth,
                 emb_size):
        self._height = height
        self._width = width
        self._depth = depth
        self._emb_size = emb_size
        super(Model, self).__init__(name, session)

    def _build(self):
        dcae = DCAE(
            'dcae',
            self._height,
            self._width,
            self._depth,
            self._emb_size
        )
        self._dcae = dcae
        #
        # Slots
        optimizer = tf.train.MomentumOptimizer(0.1, 0.618)
        optimizer = photinia.GradientClipping(optimizer, 1.0)
        update = optimizer.minimize(dcae.loss)
        self._add_slot(
            'train',
            inputs=dcae.image,
            outputs=(
                dcae.loss,
                optimizer.grad_norm
            ),
            updates=update
        )
        self._add_slot(
            'evaluate',
            inputs=dcae.image,
            outputs=dcae.rec
        )
        self._add_slot(
            'encode',
            inputs=dcae.image,
            outputs=dcae.emb
        )
        z = tf.placeholder(
            shape=(None, self._emb_size),
            dtype=photinia.D_TYPE
        )
        rec_z = dcae.decoder.setup(z)
        self._add_slot(
            'decode',
            inputs=z,
            outputs=rec_z
        )

    @property
    def dcae(self):
        return self._dcae


def main(flags):
    if os.path.exists(flags.output):
        shutil.rmtree(flags.output)
    os.mkdir(flags.output)
    #
    # Create data sources.
    ds = data.BufferedImageSource(
        flags.input,
        flags.height,
        flags.width,
        1
    )
    #
    # Create dumper.
    # dumper = photinia.FileDumper(flags.model_dir)
    dumper = photinia.TreeDumper(flags.model_dir)
    #
    # Start to train.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = Model(
            flags.name,
            session,
            flags.height,
            flags.width,
            1,
            flags.emb_size
        )
        train = model.get_slot('train')
        evaluate = model.get_slot('evaluate')
        #
        session.run(tf.global_variables_initializer())
        for i in range(1, flags.nloop + 1):
            x_batch, = ds.next_batch(flags.bsize)
            loss = train(x_batch)
            photinia.print_values(
                ['loop', 'loss', 'grad'],
                (i,) + loss
            )
            if i % 200 == 0:
                data = evaluate(x_batch)
                for j in range(20):
                    name = '{}.{}.jpg'.format(i, j)
                    name = os.path.join(flags.output, name)
                    data.save_mat(name, data[j])
                    print(name)
                    if flags.dump:
                        dumper.dump('{}@{}'.format(flags.name, i), model)
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('gpu', '0', 'Which GPU to use.')
    #
    gflags.DEFINE_integer('bsize', 32, 'Batch size.')
    gflags.DEFINE_integer('nloop', 200000, 'Max number of loop.')
    gflags.DEFINE_string('name', 'dcvae', 'Model name.')
    gflags.DEFINE_string('model_dir', 'models', 'The model path.')
    gflags.DEFINE_boolean('dump', False, 'Dump the model to model dir.')
    #
    gflags.DEFINE_string('input', 'input', 'The input data path.')
    gflags.DEFINE_string('output', 'output', 'The output path.')
    gflags.DEFINE_integer('height', 128, 'Height of the image.')
    gflags.DEFINE_integer('width', 128, 'Width of the image.')
    gflags.DEFINE_integer('emb_size', 500, 'Embedding size.')
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))
