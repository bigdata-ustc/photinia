#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-25
"""

import collections
import csv
import queue
import threading

import numpy as np


class DataSource(object):
    """DataSource
    """

    def meta(self):
        raise NotImplementedError()

    def next(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        sample = self.next()
        if sample is None:
            raise StopIteration()
        return sample


class MemorySource(DataSource):

    def __init__(self,
                 columns,
                 column_names,
                 dtype=None):
        """Construct a dataset.

        :param columns: Tuple of list, np.array or any iterable objects.
        :param dtype: Data type.
        """
        self._num_comp = len(columns)
        if self._num_comp == 0:
            raise ValueError('At least 1 data object should be given.')
        self._meta = tuple(column_name for column_name in column_names)
        self._columns = [np.array(column, dtype=dtype) for column in columns]
        size = None
        for column in self._columns:
            if size is None:
                size = len(column)
                continue
            if len(column) != size:
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

    def meta(self):
        return self._meta

    def next(self):
        if self._start >= self._size:
            self._start = 0
            self._loop += 1
            self.shuffle()
            return None
        row = tuple(
            column[self._start]
            for column in self._columns
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
            batch = tuple(self._columns[i][self._start:end].copy() for i in range(self._num_comp))
            self._start += size
        else:
            batch = tuple(self._columns[i][self._start:].copy() for i in range(self._num_comp))
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

    def __init__(self, fp, column_names, delimiter=','):
        self._meta = tuple(column_name for column_name in column_names)

        reader = csv.DictReader(fp, delimiter=delimiter)
        self._iter = iter(reader)
        self._columns = [list() for _ in self._meta]

        self._memory_source = None

    def meta(self):
        return self._meta

    def next(self):
        if self._memory_source is None:
            try:
                doc = next(self._iter)
            except StopIteration:
                self._memory_source = MemorySource(self._columns, self._meta)
                self._iter = None
                self._columns = None
                return None
            for i, column_name in enumerate(self._meta):
                cell = doc[column_name]
                self._columns[i].append(cell)
            # print('DEBUG: Fetch from file.')
            return tuple(
                column[-1]
                for column in self._columns
            )
        # print('DEBUG: Fetch from memory.')
        return self._memory_source.next()


class BatchSource(DataSource):

    def __init__(self, input_source, batch_size):
        """

        Args:
            input_source (DataSource):
            batch_size (int):

        """
        self._input_source = input_source
        self._batch_size = batch_size

        self._meta = self._input_source.meta()

        self._cell_fns = collections.defaultdict(collections.deque)
        self._column_fns = collections.defaultdict(collections.deque)

        self._eof = False

    def add_cell_fns(self, column_name, fns):
        if callable(fns):
            fns = [fns]
        elif not isinstance(fns, (list, tuple)):
            raise ValueError('fns should be callable or list(tuple) of callables.')
        if type(column_name) is not list:
            column_name = [column_name]
        for item in column_name:
            self._cell_fns[item] += fns

    def add_column_fns(self, column_name, fns):
        if callable(fns):
            fns = [fns]
        elif not isinstance(fns, (list, tuple)):
            raise ValueError('fns should be callable or list(tuple) of callables.')
        if type(column_name) is not list:
            column_name = [column_name]
        for item in column_name:
            self._column_fns[item] += fns

    def meta(self):
        return self._meta

    def next(self):
        if self._batch_size:
            if self._eof:
                self._eof = False
                return None
            columns = tuple(list() for _ in self._meta)
            for i in range(self._batch_size):
                row = self._next_one()
                if isinstance(row, Exception):
                    raise row
                if row is None:
                    if i == 0:
                        return None
                    else:
                        self._eof = True
                        break
                for j, cell in enumerate(row):
                    columns[j].append(cell)
            columns = tuple(
                self._apply_batch_mappers(column_name, column)
                for column_name, column in zip(self._meta, columns)
            )
            return columns
        else:
            return self._next_one()

    def _next_one(self):
        row = self._input_source.next()
        if row is None:
            return None
        row = tuple(
            self._apply_column_mappers(row[i], column_name)
            for i, column_name in enumerate(self._meta)
        )
        return row

    def _apply_column_mappers(self, cell, column_name):
        if column_name in self._cell_fns:
            for fn in self._cell_fns[column_name]:
                cell = fn(cell)
        return cell

    def _apply_batch_mappers(self, column_name, column):
        if column_name in self._column_fns:
            for fn in self._column_fns[column_name]:
                column = fn(column)
        return column


class ThreadBufferedSource(DataSource):

    def __init__(self, input_source, buffer_size=10000):
        self._input_source = input_source
        if isinstance(buffer_size, int) and buffer_size > 0:
            self._buffer_size = buffer_size
        else:
            raise ValueError('Argument buffer_size should be a positive integer.')
        #
        # Async Loading
        self._main_thread = threading.current_thread()
        self._queue = queue.Queue()
        self._load_threshold = self._buffer_size // 3
        self._thread = None
        self._thread_lock = threading.Semaphore(1)

    def meta(self):
        return self._input_source.meta()

    def next(self):
        if (self._queue.qsize() <= self._load_threshold
                and (self._thread is None or not self._thread.is_alive())):
            self._thread = threading.Thread(target=self._load)
            self._thread.start()
        while True:
            try:
                row = self._queue.get(block=True, timeout=1)
            except queue.Empty:
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(target=self._load)
                    self._thread.start()
                continue
            break
        if isinstance(row, Exception):
            raise row
        return row

    def _load(self):
        """This method is executed in another thread!
        """
        # print('DEBUG: Loading thread started.')
        for i in range(self._buffer_size):
            try:
                row = self._input_source.next()
            except Exception as e:
                self._queue.put(e)
                break
            self._queue.put(row, block=True)
            if not self._main_thread.is_alive():
                break
        # print('DEBUG: Loading thread stopped. %d loaded' % (i + 1))


if __name__ == '__main__':
    exit()
