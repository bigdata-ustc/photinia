#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-24
"""

import pickle
import queue
import threading

import numpy as np
import pymongo


class DataSource(object):
    """DataSource
    """

    def next_batch(self, size=0):
        """Get a batch of data.

        :param size: Batch size. Default is zero, which means extract all data.
        :return: Tuple of np.array.
        """
        raise NotImplementedError()


class Dataset(DataSource):
    """Dataset
    """

    def __init__(self,
                 *data,
                 dtype=None):
        """Construct a dataset.

        :param data: Tuple of list, np.array or any iterable objects.
        :param dtype: Data type.
        """
        self._num_comp = len(data)
        if self._num_comp == 0:
            raise ValueError('At least 1 data object should be given.')
        self._data = [np.array(mat, dtype=dtype) for mat in data]
        size = None
        for mat in self._data:
            if size is None:
                size = len(mat)
                continue
            if len(mat) != size:
                raise ValueError('All data components must have the same size.')
        self._size = size
        self._start = 0
        self._loop = 0

    @property
    def size(self):
        return self._size

    @property
    def start(self):
        return self._start

    @property
    def loop(self):
        return self._loop

    def next_batch(self, size=0):
        batch = self._next_batch(size)
        if size == 0:
            return batch
        real_size = len(batch[0])
        while real_size < size:
            batch1 = self._next_batch(size - real_size)
            batch = tuple(np.concatenate((batch[i], batch1[i]), 0) for i in range(self._num_comp))
            real_size = len(batch[0])
        return batch

    def _next_batch(self, size=0):
        if size <= 0:
            return self.all()
        if self._start == 0 and self._loop != 0:
            self.shuffle()
        end = self._start + size
        if end < self._size:
            batch = tuple(self._data[i][self._start:end].copy() for i in range(self._num_comp))
            self._start += size
        else:
            batch = tuple(self._data[i][self._start:].copy() for i in range(self._num_comp))
            self._start = 0
            self._loop += 1
        return batch

    def shuffle(self, num=3):
        perm = np.arange(self._size)
        for _ in range(num):
            np.random.shuffle(perm)
        for i in range(self._num_comp):
            self._data[i] = self._data[i][perm]
        return self

    def all(self):
        return self._data


class MongoSource(DataSource):
    """MongoDB data source
    """

    def __init__(self,
                 host, auth_db_name, user, passwd,
                 db_name,
                 coll_name,
                 match,
                 fields,
                 buffer_size):
        super(MongoSource, self).__init__()
        #
        self._host = host
        self._auth_db_name = auth_db_name
        self._user = user
        self._passwd = passwd
        self._db_name = db_name
        self._coll_name = coll_name
        self._match = match
        self._fields = fields
        self._buffer_size = buffer_size
        #
        self._project = {
            field if isinstance(field, str) else field[0]: 1
            for field in fields
        }
        self._mongo_client = pymongo.MongoClient(host)
        self._mongo_client[auth_db_name].authenticate(user, passwd)
        self._db = self._mongo_client[self._db_name]
        self._coll = self._db[self._coll_name]
        self._queue = queue.Queue()
        self._thread = None
        #
        self._one_pass_buffer = None

    def __del__(self):
        self._mongo_client.close()

    def next_batch(self, size=0):
        if size > 0:
            batch = tuple([] for _ in self._fields)
            for _ in range(size):
                if self._queue.qsize() < self._buffer_size / 3 \
                        and (self._thread is None or not self._thread.is_alive()):
                    self._thread = threading.Thread(target=self._load)
                    self._thread.start()
                doc = self._queue.get()
                if isinstance(doc, Exception):
                    raise doc
                for i, value in enumerate(doc):
                    batch[i].append(value)
            return batch
        else:
            if self._one_pass_buffer is None:
                batch = tuple([] for _ in self._fields)
                cur = self._coll.find(self._match, self._project, cursor_type=pymongo.CursorType.EXHAUST)
                for doc in cur:
                    doc = tuple(self._get_value(doc, field) for field in self._fields)
                    for i, value in enumerate(doc):
                        batch[i].append(value)
                self._one_pass_buffer = batch
            return self._one_pass_buffer

    def _load(self):
        try:
            cur = self._coll.aggregate([
                {'$match': self._match},
                {'$project': self._project},
                {'$sample': {'size': self._buffer_size}}
            ])
            for doc in cur:
                doc = tuple(self._get_value(doc, field) for field in self._fields)
                self._queue.put(doc)
        except Exception as e:
            self._queue.put(e)

    @staticmethod
    def _get_value(doc, field):
        if isinstance(field, str):
            return doc[field]
        else:
            value = doc[field[0]]
            for fn in field[1:]:
                value = fn(value)
            return value
