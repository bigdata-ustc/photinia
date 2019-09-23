#!/usr/bin/env python3


import queue
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


if __name__ == '__main__':
    l = list(range(1, 5))
    a = ThreadedAdapter(l)
    for i in a:
        print(i)
    for i in a:
        print(i)
    exit()
