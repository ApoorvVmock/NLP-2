#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive

def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    params[ofs:ofs+ Dx * H]
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Compute the probability

    h = sigmoid(np.dot(data, W1) + b1)
    y_hat = softmax(np.dot(h, W2) + b2)

    return y_hat[label]

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    gradW1 = np.zeros(shape = (len(data), Dy, H)) #todo: shape = M x Dy x H ?
    gradW2 = np.zeros(shape = (len(data), H, Dx))
    gradb1 = np.zeros(shape = (len(data), Dy, 1))
    gradb2 = np.zeros(shape = (len(data), Dy, 1))


    for i in range(len(data)):
        x, y = data[i], labels[i]
    ### YOUR CODE HERE: forward propagation
    # For all examples, we calculate accumulated errors (the cost is actually the sum of -log(forward...)
        error = -np.log(forward(x, y, params, dimensions))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # for each layer, accumulate the gradients w.r.t Weights or bias vectors. begin with W2 and b2 and then calc W1, b1

        h = sigmoid(np.dot(x, W1) + b1)
        y_hat = softmax(np.dot(h, W2) + b2)

        gradW2[i] = np.dot(h.reshape(-1, 1), y_hat-y)    # input: y (1xDy), h (1xDh). output: Dh x Dy
        gradb2[i] = y_hat - y                                # b2 (1xDy)
        gradW1[i] = np.dot()
        gradb1[i] = np.dot()

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE

    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
