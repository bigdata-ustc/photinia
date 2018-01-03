#!/usr/bin/env python3

import pickle

import numpy as np

from photinia_nlp.model import DataSource


def main():
    ds = DataSource()
    with open('/home/xi/Projects/photinia/model-1000-64/Word2vec/Emb/w.0', 'rb') as f:
        emb = pickle.load(f)
    emb /= np.sqrt(np.sum(emb ** 2, axis=1, keepdims=True))
    print(emb.shape)
    while True:
        word = input('Input a word: ')
        if word == '':
            break
        index = ds.word_to_index(word)
        if index == -1:
            print('No such word.')
            continue
        v = emb[index]
        # m = np.sum((np.reshape(v, (1, -1)) - emb) ** 2, axis=1)
        m = np.dot(v, emb.T)
        m = [(i, value) for i, value in enumerate(m)]
        m.sort(key=lambda x: -x[1])
        for i, value in m[1:6]:
            print(ds.index_to_word(i), value)
        print()
    return 0


if __name__ == '__main__':
    exit(main())
