#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-18
"""

import collections
import csv
import queue
import random
import threading
import time

import numpy as np


class DataSource(object):

    def __init__(self, field_names):
        self._field_names = list(field_names)
        self._data_model = collections.namedtuple('DataItem', field_names)

    @property
    def field_names(self):
        return self._field_names

    def next(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


class MemorySource(DataSource):

    def __init__(self,
                 field_names,
                 docs,
                 dtype=None):
        super(MemorySource, self).__init__(field_names)

        self._columns = [list() for _ in range(len(self._field_names))]
        for doc in docs:
            for i, field_name in enumerate(self._field_names):
                self._columns[i].append(doc[field_name])
        self._columns = [
            np.array(column, dtype=dtype)
            for column in self._columns
        ]

        self._num_comp = len(self._field_names)
        if self._num_comp == 0:
            raise ValueError('At least 1 data object should be given.')
        self._size = len(self._columns[0])
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

    def next(self):
        if self._start >= self._size:
            self._start = 0
            self._loop += 1
            self.shuffle()
            raise StopIteration()
        row = self._data_model(
            *(column[self._start]
              for column in self._columns)
        )
        self._start += 1
        return row

    def next_batch(self, size=0):
        batch = self._next_batch(size)
        if size == 0:
            return batch
        real_size = len(batch[0])
        while real_size < size:
            batch1 = self._next_batch(size - real_size)
            batch = self._data_model(
                *(np.concatenate((batch[i], batch1[i]), 0)
                  for i in range(self._num_comp))
            )
            real_size = len(batch[0])
        return batch

    def _next_batch(self, size=0):
        if size <= 0:
            return self.all()
        if self._start == 0 and self._loop != 0:
            self.shuffle()
        end = self._start + size
        if end < self._size:
            batch = self._data_model(
                *(self._columns[i][self._start:end].copy()
                  for i in range(self._num_comp))
            )
            self._start += size
        else:
            batch = self._data_model(
                *(self._columns[i][self._start:].copy()
                  for i in range(self._num_comp))
            )
            self._start = 0
            self._loop += 1
        return batch

    def shuffle(self, num=3):
        perm = np.arange(self._size)
        for _ in range(num):
            np.random.shuffle(perm)
        for i in range(self._num_comp):
            self._columns[i] = self._columns[i][perm]
        return self

    def all(self):
        return self._columns


class CSVSource(DataSource):

    def __init__(self, field_names, fp, delimiter=','):
        super(CSVSource, self).__init__(field_names)

        reader = csv.DictReader(fp, delimiter=delimiter)
        self._iter = iter(reader)
        self._docs = list()

        self._memory_source = None

    def next(self):
        if self._memory_source is None:
            try:
                doc = next(self._iter)
            except StopIteration:
                self._memory_source = MemorySource(self._field_names, self._docs)
                self._iter = None
                self._docs = None
                raise StopIteration()
            self._docs.append(doc)
            # print('DEBUG: Fetch from file.')
            return self._data_model(
                *(doc[field_name]
                  for field_name in self._field_names)
            )
        # print('DEBUG: Fetch from memory.')
        return self._memory_source.next()


class MongoSource(DataSource):

    def __init__(self,
                 field_names,
                 coll,
                 filters,
                 random_order,
                 buffer_size=100000):
        """Data source used to access MongoDB.

        Args:
            coll: MongoDB collection object.
            filters (dict): Filters which will be pass to MongoDB's find() operation.
            random_order (bool): If iterate the collections in random order.
                This is usually set to True when used as train set.
            buffer_size (int): Max size of the candidate buffer.
                This option will only take effect when random_order is True.

        """
        super(MongoSource, self).__init__(field_names)
        self._coll = coll
        self._filters = filters if filters is not None else {}
        self._projections = {field_name: 1 for field_name in field_names}
        self._random_order = random_order
        self._buffer_size = buffer_size

        self._cursor = None
        self._buffer = list()

    def next(self):
        if self._random_order:
            doc = self._random_next()
        else:
            doc = self._normal_order()
        return self._data_model(
            *(doc[field_name]
              for field_name in self._field_names)
        )

    def _random_next(self):
        #
        # fetch next ID from the database
        _id = None
        error = None
        for _ in range(3):
            try:
                _id = self._next_id()
                break
            except StopIteration as e:
                raise e
            except Exception as e:
                error = e
                time.sleep(3)
                continue

        #
        # add the ID from the buffer
        if _id is None:
            raise error
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(_id)
        else:
            index = random.randint(0, self._buffer_size - 1)
            self._buffer[index] = _id

        #
        # get an ID from buffer randomly
        index = random.randint(0, len(self._buffer) - 1)
        _id = self._buffer[index]

        #
        # get the doc based on the ID
        doc = None
        error = None
        for _ in range(3):
            try:
                doc = self._coll.find_one({'_id': _id}, self._projections)
                break
            except Exception as e:
                error = e
                time.sleep(3)
                continue
        if doc is None:
            raise error

        return doc

    def _next_id(self):
        if self._cursor is None:
            self._cursor = self._coll.find(self._filters, {'_id': 1})
        try:
            doc = next(self._cursor)
        except Exception as e:
            #
            # the exception may be:
            # 1) StopIteration
            # 2) CursorTimeout
            self._cursor = None
            raise e
        return doc['_id']

    def _normal_order(self):
        doc = None
        error = None
        for _ in range(3):
            if self._cursor is None:
                self._cursor = self._coll.find(self._filters, self._projections)
            try:
                doc = next(self._cursor)
            except StopIteration as e:
                raise e
            except Exception as e:
                self._cursor = None
                error = e
                time.sleep(3)
                continue
            break
        if doc is None:
            raise error

        return doc


class BatchSource(DataSource):

    def __init__(self,
                 input_source,
                 batch_size):
        """Return data item in batch.

        Args:
            input_source (DataSource): Data source to be wrapped.
            batch_size (int): Batch size.

        """
        self._input_source = input_source
        super(BatchSource, self).__init__(input_source.field_names)
        self._batch_size = batch_size

        self._cell_fns = collections.defaultdict(collections.deque)
        self._column_fns = collections.defaultdict(collections.deque)

        self._eof = False

    @property
    def batch_size(self):
        return self._batch_size

    def add_cell_fns(self, field_name, fns):
        if callable(fns):
            fns = [fns]
        elif not isinstance(fns, (list, tuple)):
            raise ValueError('fns should be callable or list(tuple) of callables.')
        if type(field_name) is not list:
            field_name = [field_name]
        for item in field_name:
            self._cell_fns[item] += fns

    def add_column_fns(self, field_name, fns):
        if callable(fns):
            fns = [fns]
        elif not isinstance(fns, (list, tuple)):
            raise ValueError('fns should be callable or list(tuple) of callables.')
        if type(field_name) is not list:
            field_name = [field_name]
        for item in field_name:
            self._column_fns[item] += fns

    def next(self):
        if self._eof:
            self._eof = False
            raise StopIteration()

        batch_doc = tuple(
            list() for _ in self._field_names
        )
        for i in range(self._batch_size):
            try:
                doc = self._next_one()
            except StopIteration as e:
                if i == 0:
                    raise e
                else:
                    self._eof = True
                    break
            for j, value in enumerate(doc):
                batch_doc[j].append(value)

        batch_doc = self._data_model(
            *(self._apply_column_fns(field_name, value)
              for field_name, value in zip(self._field_names, batch_doc))
        )
        return batch_doc

    def _next_one(self):
        doc = self._input_source.next()
        doc = tuple(
            self._apply_cell_fns(field_name, value)
            for field_name, value in zip(self._field_names, doc)
        )
        return doc

    def _apply_cell_fns(self, field_name, value):
        if field_name in self._cell_fns:
            for fn in self._cell_fns[field_name]:
                value = fn(value)
        return value

    def _apply_column_fns(self, field_name, column):
        if field_name in self._column_fns:
            for fn in self._column_fns[field_name]:
                column = fn(column)
        return column


class ThreadBufferedSource(DataSource):

    def __init__(self,
                 input_source,
                 buffer_size=1000):
        """Preload data to a buffer in another thread.

        Args:
            input_source (DataSource): Data source to be wrapped.
            buffer_size (int): Buffer size.

        """
        self._input_source = input_source
        super(ThreadBufferedSource, self).__init__(input_source.field_names)
        if isinstance(buffer_size, int) and buffer_size > 0:
            self._buffer_size = buffer_size
        else:
            raise ValueError('buffer_size should be a positive integer.')
        #
        # Async Loading
        self._queue = queue.Queue(buffer_size)
        self._thread = None

    def next(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._load)
            self._thread.setDaemon(True)
            self._thread.start()
        row = self._queue.get(block=True)
        if isinstance(row, Exception):
            raise row
        return row

    def _load(self):
        """This method is executed in another thread!
        """
        # print('DEBUG: Loading thread started.')
        while True:
            try:
                row = self._input_source.next()
            except Exception as e:
                self._queue.put(e)
                break
            self._queue.put(row, block=True)
        # print('DEBUG: Loading thread stopped. %d loaded' % (i + 1))
