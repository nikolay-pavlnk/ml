import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        width, height, channels = input_shape
        self.conv1 = ConvolutionalLayer(in_channels=3, out_channels=3, filter_size=conv1_channels)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(4, 4)
        self.conv2 = ConvolutionalLayer(in_channels=3, out_channels=3, filter_size=conv2_channels)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(4, 4)
        self.flatten = Flattener()
        self.f_connected = FullyConnectedLayer(3, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()

        params['W1'].grad = 0
        params['B1'].grad = 0
        params['W2'].grad = 0
        params['B2'].grad = 0
        params['W3'].grad = 0
        params['B3'].grad = 0
        
        Z1 = self.conv1.forward(X)
        A1 = self.relu1.forward(Z1)
        M1 = self.maxpool1.forward(A1)
        Z2 = self.conv2.forward(M1)
        A1 = self.relu2.forward(Z2)
        M2 = self.maxpool2.forward(A1)
        F = self.flatten.forward(M2)
        Out = self.f_connected.forward(F)

        loss, grad = softmax_with_cross_entropy(Out, y)
        
        grad_f_con = self.f_connected.backward(grad)
        grad_flatten = self.flatten.backward(grad_f_con)
        grad_maxpool2 = self.maxpool2.backward(grad_flatten)
        grad_relu2 = self.relu2.backward(grad_maxpool2)
        grad_conv2 = self.conv2.backward(grad_relu2)
        grad_maxpool1 = self.maxpool1.backward(grad_conv2)
        grad_relu1 = self.relu1.backward(grad_maxpool1)
        self.conv1.backward(grad_relu1)
        self.b = grad_relu1

        self.params()

        return loss

    def predict(self, X):
        Z1 = self.conv1.forward(X)
        A1 = self.relu1.forward(Z1)
        M1 = self.maxpool1.forward(A1)
        Z2 = self.conv2.forward(M1)
        A1 = self.relu2.forward(Z2)
        M2 = self.maxpool2.forward(A1)
        F = self.flatten.forward(M2)
        Out = np.argmax(self.f_connected.forward(F), axis=1)

        return Out

    def params(self):
        result = {}

        result = {
        'W2': self.conv2.W,
        'W1': self.conv1.W,
        'B2': self.conv2.B,
        'W3': self.f_connected.W,
        'B3': self.f_connected.B,
        'B1': self.conv1.B
        }


        return result
