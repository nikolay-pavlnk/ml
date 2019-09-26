import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        predictions = np.exp(predictions - np.max(predictions))
        probs = predictions / np.sum(predictions, axis=0)
    else:
        predictions = np.exp(predictions - np.max(predictions, axis=1)[:, np.newaxis])
        probs = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if len(probs.shape) == 1:
        loss = -np.log(probs[target_index])
    else:
        target_index = np.eye(probs.shape[1])[target_index]
        loss = np.sum(-np.log(probs) * target_index) / probs.shape[0]
    return loss


def softmax_with_cross_entropy(predictions, y):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, y.flatten())
    if len(probs.shape) == 1:
        probs[y] -= 1  
    else:
        m = y.shape[0]
        probs[np.arange(m), y.flatten()] -= 1
        probs /= m
        
    return loss, probs


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out * np.where(self.X > 0, 1, 0)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(np.random.normal(loc=0, scale=np.sqrt(2/(n_input+n_output)), size=(n_input, n_output)))
        self.B = Param(np.zeros(n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W.value) + self.B.value
        return self.Z

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        d_input = np.dot(d_out, self.W.value.T)
        grad_weights = np.dot(self.X.T, d_out)
        grad_biases = d_out.mean(axis=0) * self.X.shape[0]
        self.W.grad += grad_weights
        self.B.grad += grad_biases

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
