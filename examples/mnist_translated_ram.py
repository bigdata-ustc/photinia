#!/usr/bin/env python3

"""
@author: xi
@since: 2018-09-16
"""

import argparse
import os

import numpy as np
import pymongo

import photinia as ph
from photinia.apps import ram


class DataSource(ph.io.MongoSource):

    def __init__(self, coll, num_classes, random):
        self._num_classes = num_classes
        super(DataSource, self).__init__(
            coll,
            ['data', 'label_index'],
            {},
            random
        )

    def meta(self):
        return 'image', 'label'

    def next(self):
        row = super(DataSource, self).next()
        if row is None:
            return None
        data, label_index = row
        image = ph.utils.load_as_array(data, force_bgr_channels=False)
        image = np.reshape(image / 255.0, (*image.shape, 1))
        image = self.move(image)
        return image, label_index

    @staticmethod
    def move(image, size=60):
        a = np.zeros((size, size, 1), dtype=np.float32)
        xmin, xmax = 0, size - 28
        x = np.random.randint(xmin, xmax)
        y = np.random.randint(xmin, xmax)
        a[x:x + 28, y:y + 28, :] = image
        return a


def main(args):
    dataset_name = 'mnist'
    with pymongo.MongoClient('sis2.ustcdm.org') as conn:
        db = conn['images']
        coll_train = db['%s_train' % dataset_name]
        coll_test = db['%s_test' % dataset_name]

        label_set = set()
        label_set.update(coll_train.distinct('label_index'))
        label_set.update(coll_test.distinct('label_index'))
        num_classes = len(label_set)

        ds_train = DataSource(coll_train, num_classes, True)
        ds_train = ph.io.BatchSource(ds_train, args.batch_size)
        ds_train = ph.io.ThreadBufferedSource(ds_train, 100)
        ds_test = DataSource(coll_test, num_classes, False)
        ds_test = ph.io.BatchSource(ds_test, args.batch_size)

        model = ram.RAM(
            'ram',
            12, 12, 1, 3,
            128, 128, 256,
            state_size=256,
            num_classes=num_classes,
            num_steps=6,
            stddev=0.1
        )
        ph.initialize_global_variables()

        for i in range(1, args.num_loops + 1):
            batch = ds_train.next()
            if batch is None:
                batch = ds_train.next()
            image, label = batch

            loss, reward = model.train(image, label)
            if i % 50 == 0:
                print(i, 'loss:', loss, 'reward', reward)

            if i % 100 == 0:
                cal = ph.train.AccCalculator()
                for image, label in ds_test:
                    label_pred, = model.predict(image)
                    cal.update(label_pred, label)
                acc = cal.accuracy
                print()
                print('acc:', acc * 100)
                print()

    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('--batch-size', type=int, default=128)
    _parser.add_argument('--num-loops', type=int, default=50000)
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(main(_args))
