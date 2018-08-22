#!/usr/bin/env python3

"""
@author: xi
@since: 2018-02-10
"""
import io
import pickle

import numpy as np

import photinia as ph


class Vocabulary(object):
    """Vocabulary
    """

    EOS = ''

    def __init__(self, add_eos=True):
        self._add_eos = add_eos
        self._word_dict = dict()
        self._index_dict = dict()

        self._word_dict[self.EOS] = 0
        self._index_dict[0] = self.EOS
        self._voc_size = 0

    def load(self, iterable, word_field, index_field):
        """Load an existing vocabulary.

        Args:
            iterable: Iterable object. This can be a list, a generator or a database cursor.
            word_field (str): Column name that contains the word.
            index_field (str): Column name that contains the word index.

        """
        self._word_dict = {
            doc[word_field]: doc[index_field]
            for doc in iterable
        }
        self._index_dict = {
            index: word
            for word, index in self._word_dict.items()
        }
        self._voc_size = len(self._word_dict)
        return self

    def generate(self, iterable, words_field):
        """Generate a vocabulary from sentences.

        Args:
            iterable: Iterable object. This can be a list, a generator or a database cursor.
            words_field (str): Column name that contains "words" data.

        """
        for doc in iterable:
            words = doc[words_field]
            for word in words:
                if word not in self._word_dict:
                    self._word_dict[word] = len(self._word_dict)
        self._index_dict = {
            index: word
            for word, index in self._word_dict.items()
        }
        self._voc_size = len(self._word_dict)
        return self

    @property
    def voc_size(self):
        return self._voc_size

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def index_dict(self):
        return self._index_dict

    def word_to_one_hot(self, word):
        return ph.utils.one_hot(self._word_dict[word], self._voc_size, np.float32)

    def one_hot_to_word(self, one_hot):
        index = np.argmax(one_hot)
        try:
            return self._index_dict[index]
        except KeyError:
            raise ValueError('Index %d is not in vocabulary.' % index)

    def words_to_one_hots(self, words):
        one_hot_list = [
            ph.utils.one_hot(self._word_dict[word], self._voc_size, np.float32)
            for word in words
        ]
        if self._add_eos:
            one_hot_list.append(ph.utils.one_hot(0, self._voc_size, np.float32))
        return one_hot_list

    def one_hots_to_words(self, one_hots):
        with io.StringIO() as buffer:
            for one_hot in one_hots:
                index = np.argmax(one_hot)
                try:
                    word = self._index_dict[index]
                    if word == '':
                        break
                    buffer.write(word)
                except KeyError:
                    raise ValueError('Index %d is not in vocabulary.' % index)
            return buffer.getvalue()


class WordEmbedding(object):

    def __init__(self,
                 mongo_coll,
                 word_field='word',
                 vec_field='vec'):
        self._coll = mongo_coll
        self._word_field = word_field
        self._vec_field = vec_field
        #
        self._word_dict = {}

    def get_vector(self, word, emb_size=None):
        if word not in self._word_dict:
            vec = self._coll.find_one({self._word_field: word}, {self._vec_field: 1})
            if vec is None:
                self._word_dict[word] = None if emb_size is None else np.random.normal(0, 1.0, emb_size)
            else:
                self._word_dict[word] = pickle.loads(vec[self._vec_field])
        return self._word_dict[word]

    def words_to_vectors(self,
                         words,
                         delimiter=None,
                         lowercase=True,
                         emb_size=None):
        """Convert a sentence into word vector list.

        :param words: A string or a list of string.
        :param delimiter: If "words" is a string, delimiter can be used to split the string into word list.
        :param lowercase: If the words be converted into lower cases during the process.
        :param emb_size: integer. Embedding size.
        :return: A list of vectors.
        """
        if delimiter is not None:
            words = words.split(delimiter)
        if lowercase:
            words = [word.lower() for word in words]
        vectors = np.array([
            vec for vec in (self.get_vector(word, emb_size) for word in words)
            if vec is not None
        ], dtype=np.float32)
        return vectors


def pad_sequences(array_list, dtype=np.float32):
    batch_size = len(array_list)
    seq_len = max(map(len, array_list))
    word_size = len(array_list[0][0])
    ret = np.zeros((batch_size, seq_len, word_size), dtype=dtype)
    for i, arr in enumerate(array_list):
        for j, row in enumerate(arr):
            ret[i, j] = row
    return ret
