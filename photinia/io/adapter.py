#!/usr/bin/env python3
import collections
import queue
import random
import threading


class Adapter(object):

    def __init__(self):
        self.__self_iter__ = None

    def __iter__(self):
        raise NotImplementedError()

    def next(self):
        if self.__self_iter__ is None:
            self.__self_iter__ = iter(self)
        try:
            return next(self.__self_iter__)
        except StopIteration as e:
            self.__self_iter__ = None
            raise e


class ThreadedAdapter(Adapter):

    def __init__(self,
                 iterable,
                 fn=None,
                 buffer_size=8192,
                 auto_reload=False):
        super(ThreadedAdapter, self).__init__()
        self._iterable = iterable
        self._fn = fn
        self._buffer_size = buffer_size
        self._auto_reload = auto_reload

        self._buffer = queue.Queue(maxsize=buffer_size)
        self._thread = None

    def __iter__(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._load_thread, daemon=True)
            self._thread.start()
        return self

    def _load_thread(self):
        while True:
            for doc in self._iterable:
                if callable(self._fn):
                    doc = self._fn(doc)
                if doc is not None:
                    self._buffer.put(doc)
            # append None to indicate stop iteration
            self._buffer.put(None)
            # if not in "auto reload" mode, a new thread will be create in the next iteration
            if not self._auto_reload:
                self._thread = None
                break

    def __next__(self):
        doc = self._buffer.get()
        if doc is None:
            raise StopIteration()
        return doc


class BatchAdapter(Adapter):

    def __init__(self, iterable, batch_size):
        super(BatchAdapter, self).__init__()
        self._iterable = iterable
        self._batch_size = batch_size
        self._it = None

    def __iter__(self):
        self._it = iter(self._iterable)
        return self

    def __next__(self):
        if self._it is None:
            raise StopIteration()
        batch = collections.defaultdict(list)
        for i in range(self._batch_size):
            try:
                doc = next(self._it)  # type: dict
            except StopIteration:
                if i == 0:
                    raise StopIteration()
                else:
                    self._it = None
                    break
            for k, v in doc.items():
                batch[k].append(v)
        return batch


class ShuffledAdapter(Adapter):

    def __init__(self, iterable):
        super(ShuffledAdapter, self).__init__()
        self._item_list = list(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        return random.choice(self._item_list)


if __name__ == '__main__':
    l = list(range(1, 5))
    a = ThreadedAdapter(l)
    for i in a:
        print(i)
    for i in a:
        print(i)
    exit()
