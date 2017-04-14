#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                      index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0

    for seq in dataset:
        for i in range(2, len(seq)):
            curr, prev, prevprev = seq[i], seq[i - 1], seq[i - 2]
            if (curr, prev, prevprev) not in trigram_counts:
                trigram_counts[(curr, prev, prevprev)] = 0
            trigram_counts[(curr, prev, prevprev)] += 1

            if (curr, prev) not in bigram_counts:
                bigram_counts[(curr, prev)] = 0
            bigram_counts[(curr, prev)] += 1

            if curr not in unigram_counts:
                unigram_counts[curr] = 0
            unigram_counts[curr] += 1

    token_count += sum(unigram_counts.values())

    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    l = 0
    lambda3 = 1. - lambda1 - lambda2
    for seq in eval_dataset:
        for i in range(2, len(seq)):
            curr, prev, prevprev = seq[i], seq[i - 1], seq[i - 2]
            if ((prev, prevprev) in bigram_counts) and ((curr, prev, prevprev) in trigram_counts):
                tri_prob = float(trigram_counts[(curr, prev, prevprev)]) / bigram_counts[(prev, prevprev)]
            else:
                tri_prob = 0

            if ((curr, prev) in bigram_counts) and (prev in unigram_counts):
                bi_prob = float(bigram_counts[(curr, prev)]) / unigram_counts[prev]
            else:
                bi_prob = 0

            if curr in unigram_counts:
                uni_prob = float(unigram_counts[curr]) / train_token_count
            else:
                uni_prob = 0

            p_seq = (lambda1 * tri_prob) + (lambda2 * bi_prob) + (lambda3 * uni_prob)
            l += np.log2(p_seq)

    l = float(l) / train_token_count
    print l
    perplexity += 2 ** (-l)

    return perplexity

def grid_search_lambdas(trigram_counts, bigram_counts, unigram_counts, token_count):
    perplexities = np.zeros(shape=(100,100))
    for i,lambda1 in enumerate(np.arange(0,1,0.01)):
        for j,lambda2 in enumerate(np.arange(0,1,0.01)):
            perplexities[i][j] = evaluate_ngrams(
                S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)

    min_i, min_j = np.unravel_index(perplexities.argmin(), perplexities.shape)
    return "min lambda1: "+np.arange(0,1,0.01)[min_i]+" min lambda2: "+np.arange(0,1,0.01)[min_j]

def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    # Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    # perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    # grid_search_lambdas()
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    test_ngram()
