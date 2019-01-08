#!/usr/bin/env python3

"""
@author: xi
@since: 2018-02-10
"""
import collections
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
        self._word_dict = None
        self._word_list = None
        self._voc_size = None

    def load(self, iter_voc_item, word_column='word', index_column='index'):
        """Load an existing vocabulary.

        Args:
            iter_voc_item: Iterable object. This can be a list, a generator or a database cursor.
            word_column (str): Column name that contains the word.
            index_column (str): Column name that contains the word index.

        """
        # load word_dict
        word_dict = dict()
        for doc in iter_voc_item:
            word = doc[word_column]
            index = doc[index_column]
            word_dict[word] = index

        # generate word_list
        voc_size = len(word_dict)
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        self._word_dict = word_dict
        self._word_list = word_list
        self._voc_size = voc_size
        return self

    def dump(self, word_column='word', index_column='index'):
        """Dump the current vocabulary to a dict generator.

        Args:
            word_column (str): Column name for word.
            index_column (str): Column name for index.

        Returns:
            A generator of dict object.

        """
        for word, index in self._word_dict.items():
            yield {
                word_column: word,
                index_column: index
            }

    def generate(self, iter_words, words_column='words', min_count=1, verbose_fn=None):
        """Generate a vocabulary from sentences.

        Args:
            iter_words: Iterable object. This can be a list, a generator or a database cursor.
            words_column (str): Column name that contains "words" data.
            min_count (int): Minimum count of the word in the vocabulary.
            verbose_fn ((int) -> None): Verbose function.
                This is useful when iter_words contains much more documents.

        """
        # statistic info
        counter = collections.defaultdict(int)
        for i, doc in enumerate(iter_words, 1):
            words = doc[words_column]
            for word in words:
                counter[word] += 1
            if verbose_fn:
                verbose_fn(i)
        if '' in counter:
            del counter['']

        # generate word_dict (word -> index)
        word_dict = {self.EOS: 0}
        for word, count in counter.items():
            if count < min_count:
                continue
            index = len(word_dict)
            word_dict[word] = index

        # generate word_list
        voc_size = len(word_dict)
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        self._word_dict = word_dict
        self._word_list = word_list
        self._voc_size = voc_size
        return self

    @property
    def voc_size(self):
        return self._voc_size

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def word_list(self):
        return self._word_list

    def word_to_one_hot(self, word):
        if word in self._word_dict:
            return ph.utils.one_hot(self._word_dict[word], self._voc_size, np.float32)
        else:
            return None

    def one_hot_to_word(self, one_hot):
        index = np.argmax(one_hot)
        try:
            return self._word_list[index]
        except KeyError:
            raise ValueError('Index %d is not in vocabulary.' % index)

    def words_to_indexes(self, words):
        index_list = [
            self._word_dict[word]
            for word in words
            if word in self._word_dict
        ]
        if self._add_eos:
            index_list.append(0)
        return index_list

    def words_to_one_hots(self, words):
        one_hot_list = [
            ph.utils.one_hot(self._word_dict[word], self._voc_size, np.float32)
            for word in words
            if word in self._word_dict
        ]
        if self._add_eos:
            one_hot_list.append(ph.utils.one_hot(0, self._voc_size, np.float32))
        return one_hot_list

    def indexes_to_words(self, indexes):
        with io.StringIO() as buffer:
            for index in indexes:
                try:
                    word = self._word_list[index]
                    if word == '':
                        break
                    buffer.write(word)
                except KeyError:
                    raise ValueError('Index %d is not in the vocabulary.' % index)
            return buffer.getvalue()

    def one_hots_to_words(self, one_hots):
        with io.StringIO() as buffer:
            for one_hot in one_hots:
                index = int(np.argmax(one_hot))
                try:
                    word = self._word_list[index]
                    if word == '':
                        break
                    buffer.write(word)
                except KeyError:
                    raise ValueError('Index %d is not in the vocabulary.' % index)
            return buffer.getvalue()


class WordEmbedding(object):

    def __init__(self):
        self._word_dict = None
        self._word_list = None
        self._emb_mat = None

    def load(self, iter_emb_item, word_column='word', index_column='index', vector_column='vector'):
        # load word_dict and emb_dict
        word_dict = dict()
        emb_dict = dict()
        for doc in iter_emb_item:
            word = doc[word_column]
            index = doc[index_column]
            vector = doc[vector_column]
            word_dict[word] = index
            emb_dict[index] = vector
        voc_size = len(word_dict)

        # generate word_list
        word_list = [None for _ in range(voc_size)]
        for word, index in word_dict.items():
            word_list[index] = word

        # generate emb_list
        emb_list = [None for _ in range(voc_size)]
        for index, vector in emb_dict.items():
            emb_list[index] = vector

        self._word_dict = word_dict
        self._word_list = word_list
        self._emb_mat = np.array(emb_list, np.float32)
        return self

    def dump(self, word_column='word', index_column='index', vector_column='vector'):
        """Dump the current vocabulary to a dict generator.

        Args:
            word_column (str): Column name for word.
            index_column (str): Column name for index.
            vector_column (str): Column name for vector.

        Returns:
            A generator of dict object.

        """
        for word, index in self._word_dict.items():
            vector = self._emb_mat[index]
            yield {
                word_column: word,
                index_column: index,
                vector_column: pickle.dumps(vector)
            }

    def generate(self,
                 voc,
                 iter_pre_trained,
                 word_column='word',
                 vector_column='vector',
                 bound=(-1.0, 1.0),
                 verbose_fn=None):
        """Generate word embedding.

        Args:
            voc (Vocabulary): Vocabulary.
            iter_pre_trained: Iterator/Generator of per-trained word2vec.
            word_column (str): Column name for word.
            vector_column (str): Column name for vector.
            bound (tuple[float]): Bound of the uniform distribution which is used to generate vectors for words that
                not exist in pre-trained word2vec.
            verbose_fn ((int) -> None): Verbose function to indicate progress.

        """
        # inherit input vocabulary's word_dict and word_list
        self._word_dict = voc.word_dict
        self._word_list = voc.word_list

        # generate emb_list
        emb_size = None
        emb_list = [None for _ in range(voc.voc_size)]  # type: list
        for i, doc in enumerate(iter_pre_trained, 1):
            if verbose_fn:
                verbose_fn(i)
            word = doc[word_column]
            vector = doc[vector_column]
            if emb_size is None:
                emb_size = len(vector)
            try:
                index = self._word_dict[word]
            except KeyError:
                continue
            emb_list[index] = vector

        # generate random vectors
        for i, vector in enumerate(emb_list):
            vector = emb_list[i]
            if vector is None:
                vector = np.random.uniform(bound[0], bound[1], emb_size)
            emb_list[i] = vector

        self._emb_mat = np.array(emb_list, np.float32)
        return self

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def word_list(self):
        return self._word_list

    @property
    def emb_mat(self):
        return self._emb_mat


class WordEmbedding1(object):

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


def pad_sequences(sequences, padding=None, axis=1, max_length=None):
    """Pad a batch of sequences.
    Default shape of the input: (batch_size, seq_length, ...)

    Args:
        sequences (list): A list represents a a sequence batch.
            Here the input "sequences" must be a list since sequences in it may have different length.
            If we regard the list as a tensor, the first dimension must be "batch_size" and the second dimension
            must be "sequence length".
            For example:
                before = [
                    [1, 3, 5, 3, 1, 0],
                    [3, 8, 3, 0],
                    [1, 3, 3, 1, 4, 4, 7, 5, 0]
                ]
                after = pad_sequences(before) = [
                    [1, 3, 5, 3, 1, 0, 0, 0, 0],
                    [3, 8, 3, 0, 0, 0, 0, 0, 0],
                    [1, 3, 3, 1, 4, 4, 7, 5, 0]
                ]
        padding: Element used to pad the sequence.
            The default padding is a zero tensor with the same shape as the original element.
        axis (int): The sequence axis.
            For example:
                x = [
                    [1, 3, 5, 3, 1, 0],
                    [3, 8, 3, 0],
                    [1, 3, 3, 1, 4, 4, 7, 5, 0]
                ]
            The axis for x is "1".
        max_length (int): The length limit of the sequences.

    Returns:
        list: The list represents a batch of sequence.

    """
    # if axis > 1:
    #     return [
    #         pad_sequences(seq, padding, axis - 1)
    #         for seq in sequences
    #     ]
    # elif axis == 1:
    #     if padding is None:
    #         padding = np.zeros_like(sequences[0][0])
    #     info_list = [(len(seq), seq) for seq in sequences]
    #     new_len = max(info_list, key=lambda a: a[0])[0]
    #     if max_length is not None and new_len > max_length:
    #         new_len = max_length
    #     return [
    #         [*(np.array(elem) for i, elem in enumerate(seq) if i < new_len),
    #          *(padding for _ in range(new_len - seq_len))]
    #         for seq_len, seq in info_list
    #     ]
    # else:
    #     raise ValueError('axis should be larger than 0.')
    max_len = _get_max_len(sequences, axis)
    if max_length is not None and max_len > max_length:
        max_len = max_length
    return _pad_seq(sequences, padding, axis, max_len)


def _get_max_len(seq, axis):
    if axis > 0:
        return max(_get_max_len(elem, axis - 1) for elem in seq)
    return len(seq)


def _pad_seq(seq, padding, axis, max_len):
    if axis > 1:
        return [
            _pad_seq(elem, padding, axis - 1, max_len)
            for elem in seq
        ]
    elif axis == 1:
        if padding is None:
            padding = np.zeros_like(seq[0][0])
        return [
            [*(np.array(elem_i) for i, elem_i in enumerate(elem) if i < max_len),
             *(padding for _ in range(max_len - len(elem)))]
            for elem in seq
        ]
    else:
        raise ValueError('axis should be larger than 0.')
