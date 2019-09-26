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
    # TODO copy from the previous assignment
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
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

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding=0):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        np.random.seed(42)
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def get_convolution(self, X, W):
        batch_size, height, width, channels = X.shape
        return np.dot(X.reshape((batch_size, height*width*channels)), W) + self.B.value

    def forward(self, X):
        # N, H, W, C = X.shape
        # HH, WW, _, F = self.W.value.shape
        # stride, pad = 1, 0
        # H_out = int(1 + (H + 2 * pad - HH) // stride)
        # W_out = int(1 + (W + 2 * pad - WW) // stride)
        # out = np.zeros((N, H_out, W_out, F))
        # x_pad = np.pad(X, pad_width=((0,0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
        # Npad, Hpad, Wpad, Cpad = x_pad.shape
        # for n in range(N):
        #     for c in range(F):
        #         for x in range(0, Hpad - (HH - 1), stride):
        #             for y in range(0, Wpad - (WW - 1), stride):
        #                 prod = np.sum(np.multiply(self.W.value[:, :, :, c], x_pad[n, x:x+HH, y:y+WW, :]))
        #                 out[n, int(x/stride), int(y/stride), c] = prod + self.B.value[c]
        # self.X = x_pad
        # return out
        new_height = int(((X.shape[1] - self.filter_size + 2 * self.padding) / 1) + 1)
        new_width = int(((X.shape[2] - self.filter_size + 2 * self.padding) / 1) + 1)
        Z = np.zeros((X.shape[0], new_height, new_width, self.out_channels))

        self.X_prev_shape = X.shape
        X = np.pad(X, pad_width=((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        W_reshaped = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels)

        _, height, width, _ = X.shape
        for x in np.arange(new_height):
            for y in np.arange(new_width):
                Z[:, x, y, :] = self.get_convolution(X[:, x:x+self.filter_size, y:y+self.filter_size, :], W_reshaped)

        self.X = X
        return Z

    def backward(self, d_out):
        # N, H, W, C = self.X.shape
        # HH, WW, _, F = self.W.value.shape
        # _, Hout, Wout, _ = d_out.shape
        # pad, stride = 0, 1
        # x_pad = self.X.copy()
        # dxpad = np.zeros_like(x_pad)
        # dw = np.zeros_like(self.W.value)
        # db = np.zeros(F)
        # for n in range(N):
        #     for c in range(F):
        #         db[c] += np.sum(d_out[n, :, :, c])
        #         for x in range(Hout):
        #             for y in range(Wout):
        #                 dw[:, :, :, c] += x_pad[n, x*stride:x*stride+HH, y*stride:y*stride+WW, :] * d_out[n, x, y, c]
        #                 dxpad[n, x*stride:x*stride+HH, y*stride:y*stride+WW, :] += self.W.value[:, :, :, c] * d_out[n, x, y, c]
        # dx = dxpad[:, pad:pad+H, pad:pad+W, :]

        # self.W.grad = dw.copy()
        # self.B.grad = db.copy()
        # return dx

        batch_size, height, width, channels = self.X.shape

        d_input = np.zeros((self.X.shape))
        W_reshaped = self.W.value.reshape((self.filter_size*self.filter_size*self.in_channels, self.out_channels))
        
        for x in np.arange(d_out.shape[1]):
            for y in np.arange(d_out.shape[2]):
                x_slice = self.X[:, x:x+self.filter_size, y:y+self.filter_size, :]
                grad_slice = d_out[:, x, y, :]
                grad_weights = np.dot(x_slice.reshape((batch_size, self.filter_size*self.filter_size*channels)).T, grad_slice)
                self.W.grad += grad_weights.reshape((self.filter_size, self.filter_size, self.in_channels, self.out_channels))
                d_input[:, x:x+self.filter_size, y:y+self.filter_size, :] += np.dot(grad_slice, W_reshaped.T).reshape((batch_size, self.filter_size, self.filter_size, self.in_channels))
        
        self.B.grad = np.mean(np.sum(np.sum(d_out, axis=1), axis=1), axis=0) * batch_size

        if self.padding:
           d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]
   
        return d_input


    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        new_height = int(((height - self.pool_size) / self.stride) + 1)
        new_width = int(((width - self.pool_size) / self.stride) + 1)
        Z = np.zeros((batch_size, new_height, new_width, channels))

        # for x in range(new_height):
        # 	for y in range(new_width):
        # 		Z[:, x, y, :] = np.max(X[:, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size, :].reshape((batch_size, self.pool_size*self.pool_size, channels)), axis=1)
        # self.X = X
        # return Z
        for n in range(batch_size):
            for c in range(channels):
                for x in range(new_height):
                    for y in range(new_width):
                        Z[n, x, y, c] = np.max(X[n, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size, c])
        self.X = X
        return Z

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        d_input = np.zeros((batch_size, height, width, channels))
        
        for n in range(batch_size):
            for c in range(channels):
                for x in range(d_out.shape[1]):
                    for y in range(d_out.shape[2]):
                        x_pool = self.X[n, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size, c]
                        mask = (x_pool == np.max(x_pool))
                        d_input[n, x*self.stride:x*self.stride+self.pool_size, y*self.stride:y*self.stride+self.pool_size, c] = mask * d_out[n, x, y, c]
        
        return d_input
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape

        return X.reshape((batch_size, height*width*channels))

    def backward(self, d_out):
        batch_size, height, width, channels = self.X_shape
        return d_out.reshape((batch_size, height, width, channels))

    def params(self):
        return {}
