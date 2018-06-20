#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-20
"""

import collections
import random


class ReplayMemory(object):

    def __init__(self, buffer_size):
        """Replay memory.

        Args:
            buffer_size (int): Max buffer size.

        """
        self._buffer_size = buffer_size
        self._buffer = collections.deque()

    def full(self):
        return len(self._buffer) >= self._buffer_size

    def put(self, s, a, r, s_, done):
        """Put a transition tuple to the replay memory.

        Args:
            s (numpy.ndarray): State s_t.
            a ((numpy.ndarray)): Action a_t.
            r (float): Reward r_{t + 1}.
            s_ (numpy.ndarray): Transition state s_{t + 1}.
            done (bool): Is terminal?

        """
        self._buffer.append((s, a, r, s_, done))
        if len(self._buffer) > self._buffer_size:
            self._buffer.popleft()

    def get(self, batch_size):
        """Get a random batch of transitions from the memory.

        Args:
            batch_size (int): Batch size.

        Returns:
            list[tuple]: List of transition tuples.

        """
        columns = (list(), list(), list(), list(), list())
        rows = random.sample(list(self._buffer), batch_size) if batch_size <= len(self._buffer) else self._buffer
        for row in rows:
            # for col in row:
            #     print(col)
            for i in range(5):
                columns[i].append(row[i])
        return columns
