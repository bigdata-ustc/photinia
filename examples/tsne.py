#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对word embedding使用TSNE进行降维并可视化

@author: winton 
@time: 2017-10-27 10:42 
"""
import sys

import gflags
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    x = low_dim_embs[:, 0]
    y = low_dim_embs[:, 1]
    plt.scatter(x, y)
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    # plt.savefig(filename)


def main(flags):
    with open(flags.emb_file, 'rb') as f:
        emb_dict = pickle.load(f)
    final_embeddings = []
    words = []
    for k, v in emb_dict.items():
        words.append(k)
        final_embeddings.append(v)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(final_embeddings[:flags.plot_num])
    labels = words[:flags.plot_num]
    plot_with_labels(low_dim_embs, labels)
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('emb_file', './embedding/embedding_20000.pkl', 'Word embedding file.')
    gflags.DEFINE_integer('plot_num', 200, 'Number of word to show.')
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    exit(main(global_flags))
