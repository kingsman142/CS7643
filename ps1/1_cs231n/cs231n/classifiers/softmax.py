import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = W.shape[0]
  N = X.shape[1]

  # calculate softmax scores
  z = np.matmul(W, X) # z = Wx + b (b is not passed into this function) -- CxN
  num = np.exp(z - np.max(z, axis=0)) # numerator of softmax -- CxN matrix with element-wise exponentiation (e^x) of z
  den = np.sum(num, axis=0) # denominator of softmax -- 1xN matrix with sum of each row
  softmax = np.divide(num, den) # carry out softmax calculation -- CxN

  # calculate regularization and loss
  regularization = np.sum(W*W) # sum of squares of weight matrix W
  loss = - np.sum(np.log(softmax[y, range(N)])) / N + regularization # Loss = L(W) + R(W)

  # calculate gradients of weights
  softmax[y, range(N)] -= 1.0 # derivative of softmax is subtracting 1.0 from the weight of the correct class of each sample
  softmax /= N
  dW = np.dot(softmax, np.transpose(X))
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
