import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.first_activation = ReLULayer()
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        params = self.params()

        params['W1'].grad = 0
        params['B1'].grad = 0
        params['W2'].grad = 0
        params['B2'].grad = 0

        Z1 = self.first_layer.forward(X)
        A1 = self.first_activation.forward(Z1)
        Z2 = self.second_layer.forward(A1)

        loss, grad = softmax_with_cross_entropy(Z2, y)
        
        grad_2 = self.second_layer.backward(grad)
        grad_activation_1 = self.first_activation.backward(grad_2)
        grad_1 = self.first_layer.backward(grad_activation_1)

        params = self.params()

        loss_reg, grad_reg = l2_regularization(params['W1'].value, self.reg)
        loss += loss_reg
        params['W1'].grad += grad_reg

        loss_reg, grad_reg = l2_regularization(params['W2'].value, self.reg)
        loss += loss_reg
        params['W2'].grad += grad_reg

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        Z1 = self.first_layer.forward(X)
        A1 = self.first_activation.forward(Z1)
        Z2 = np.argmax(self.second_layer.forward(A1), axis=1)

        return Z2

    def params(self):
        result = {'W1': self.first_layer.W,
        'B1': self.first_layer.B,
        'W2': self.second_layer.W,
        'B2': self.second_layer.B
        }

        return result
