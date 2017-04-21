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
        curr, prev = seq[1], seq[0]
        bigram_counts[(curr, prev)] = bigram_counts.get((curr, prev), 0) + 1
        unigram_counts[curr] = unigram_counts.get(curr, 0) + 1
        unigram_counts[prev] = unigram_counts.get(prev, 0) + 1
        token_count += 2
        for i in range(2, len(seq)):
            curr, prev, prevprev = seq[i], seq[i - 1], seq[i - 2]
            trigram_counts[(curr, prev, prevprev)] = trigram_counts.get((curr, prev, prevprev), 0) + 1
            bigram_counts[(curr, prev)] = bigram_counts.get((curr, prev), 0) + 1
            unigram_counts[curr] = unigram_counts.get(curr, 0) + 1
            token_count += 1

    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    l = 0
    test_token_count = 0
    lambda3 = round(1. - lambda1 - lambda2, 2)  # Avoids floating point inaccuracies
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

            uni_prob = float(unigram_counts.get(curr, 0)) / train_token_count

            p_seq = (lambda1 * tri_prob) + (lambda2 * bi_prob) + (lambda3 * uni_prob)
            if p_seq == 0:
                return float("inf")
            l += np.log2(p_seq)
            test_token_count += 1

    l = float(l) / test_token_count
    perplexity += 2 ** (-l)

    return perplexity


def grid_search_lambdas(trigram_counts, bigram_counts, unigram_counts, token_count):
    perplexities = np.zeros(shape=(101, 101))
    iter_counter = 0
    opt_perplexity = float("inf")
    opt_lambda1 = 0
    opt_lambda2 = 0
    for lambda1 in np.arange(0, 1.01, 0.01):
        for lambda2 in np.arange(0, 1.01-lambda1, 0.01):
            perplexity = evaluate_ngrams(
                S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)
            if perplexity < opt_perplexity:
                opt_perplexity = perplexity
                opt_lambda1 = lambda1
                opt_lambda2 = lambda2
                print("Reached new low! : " + str(perplexity))

            print(str(iter_counter) + ", for lambda1 = " + str(lambda1) +
                  ", lambda2 = " + str(lambda2) +
                  ", lambda3 = " + str(round(1.-lambda1-lambda2, 2)) +
                    ", perplexity is " + str(perplexity))
            iter_counter += 1

    opt_lambda_3 = 1 - opt_lambda1 - opt_lambda2
    print "best lambda1: " + str(opt_lambda1) +\
           " best lambda2: " + str(opt_lambda2) +\
           " best lambda3: " + str(opt_lambda_3) +\
            " perplexity: " + str(opt_perplexity)


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
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.37, 0.5)
    print "#perplexity: " + str(perplexity)
    grid_search_lambdas(trigram_counts, bigram_counts, unigram_counts, token_count)

    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    test_ngram()
