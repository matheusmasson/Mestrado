
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, softmax, sigmoid, ScaledTanH, linear
from lasagne.updates import nesterov_momentum
from lasagne.layers import get_output, Upscale2DLayer, ReshapeLayer


from lasagne.layers import get_all_params
import lasagne as lsg


from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
from lasagne.layers.cuda_convnet import NINLayer_c01b
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion


def get_network(shape1, shape2, net_num):

    net_ret = []

    if net_num == 1:

        net_ret = [
            # layer dealing with the input data
            (InputLayer, {'shape': (None, 1, shape1, shape2)}),

           (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),
             (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),
             (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),

            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

            # first stage of our convolutional layers

             (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),
             (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),

            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

            # second stage of our convolutional layers
            (Conv2DLayerFast, {'num_filters': 64, 'filter_size': (3, 3)}),
            (Conv2DLayerFast, {'num_filters': 64, 'filter_size': (3, 3)}),

            #(NINLayer_c01b, {'num_units': 32}),
            (MaxPool2DLayerFast, {'pool_size':  (2, 2)}),

             # third stage of our convolutional layers
            (Conv2DLayerFast, {'num_filters': 128, 'filter_size': (3, 3)}),
                (Conv2DLayerFast, {'num_filters': 128, 'filter_size': (3, 3)}),

            #(NINLayer_c01b, {'num_units': 32}),
            (MaxPool2DLayerFast, {'pool_size':  (2, 2)}),



             # four stage of our convolutional layers
            (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),

            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),



            # two dense layers with dropout
            (DenseLayer, {'num_units': 64}),
            (DropoutLayer, {}),
            (DenseLayer, {'num_units': 64}),

            # the output layer
            (DenseLayer, {'num_units': 50, 'nonlinearity': tanh}),
        ]

    elif net_num == 2:
        net_ret = [
            # layer dealing with the input data
            (InputLayer, {'shape': (None, shape1, shape2)}),

            (lsg.layers.Conv1DLayer, {'num_filters': 32, 'filter_size': 3}),
            (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

            (lsg.layers.Conv1DLayer, {'num_filters': 64, 'filter_size': 3}),
            (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

            (lsg.layers.Conv1DLayer, {'num_filters': 128, 'filter_size': 3}),
             (lsg.layers.Conv1DLayer, {'num_filters': 128, 'filter_size': 3}),
            (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

            (lsg.layers.Conv1DLayer, {'num_filters': 32, 'filter_size': 3}),
            (lsg.layers.Conv1DLayer, {'num_filters': 32, 'filter_size': 3}),
            (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

            (lsg.layers.Conv1DLayer, {'num_filters': 16, 'filter_size': 3}),
            (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

            # two dense layers with dropout
            (DenseLayer, {'num_units': 128}),
            (DropoutLayer, {}),
            (DenseLayer, {'num_units': 128}),

            # the output layer
            (DenseLayer, {'num_units': 50, 'nonlinearity': tanh}),
        ]
    elif net_num == 3:

        net_ret = [

            (InputLayer, {'shape': (None, 1, shape1, shape2)}),

            (Conv2DLayerFast, {'num_filters': 16, 'filter_size': 3}),
            (Conv2DLayerFast, {'num_filters': 16, 'filter_size': 3}),
            (MaxPool2DLayerFast, {'pool_size': 2}),

            (Conv2DLayerFast, {'num_filters': 32, 'filter_size': 3}),
            (MaxPool2DLayerFast, {'pool_size': 2}),

            (ReshapeLayer, {'shape': (([0], -1))}),
            (DenseLayer, {'num_units': 128}),
            (DenseLayer, {'name': 'encode', 'num_units': 16}),
            (DenseLayer, {'num_units': 128}),
            (DenseLayer, {'num_units': 800}),
            (ReshapeLayer, {'shape': (([0], 32, 5, 5))}),
            (Upscale2DLayer, {'scale_factor': 2}),

            (Conv2DLayerFast, {'num_filters': 16, 'filter_size': 3}),
            (Upscale2DLayer, {'scale_factor': 2}),
            (Conv2DLayerFast, {'num_filters': 16, 'filter_size': 3}),

            (MaxPool2DLayerFast, {'pool_size': 2}),


            (lsg.layers.Conv2DLayer, {'num_filters': 2, 'filter_size': 3}),
            (ReshapeLayer, {'shape': (([0], -1))}),
         ]

    elif net_num == 4:

        net_ret = [

            (InputLayer, {'shape': (None, 1, shape1, shape2)}),

            (Conv2DLayerFast, {'num_filters': 16, 'filter_size': 3}),
            (Conv2DLayerFast, {'num_filters': 16, 'filter_size': 3}),
            (MaxPool2DLayerFast, {'pool_size': 2}),

            (Conv2DLayerFast, {'num_filters': 32, 'filter_size': 3}),
            (MaxPool2DLayerFast, {'pool_size': 2}),

            (ReshapeLayer, {'shape': (([0], -1))}),
            (DenseLayer, {'num_units': 128}),
            (DenseLayer, {'name': 'encode', 'num_units': 16}),
            (DenseLayer, {'num_units': 128}),
            (DenseLayer, {'num_units': 800}),
            (ReshapeLayer, {'shape': (([0], 32, 5, 5))}),
            (Upscale2DLayer, {'scale_factor': 2}),

            (Conv2DLayerFast, {'num_filters': 16, 'filter_size': 3}),
            (Upscale2DLayer, {'scale_factor': 2}),
            (Conv2DLayerFast, {'num_filters': 16, 'filter_size': 3}),

            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

            # two dense layers with dropout
            (DenseLayer, {'num_units': 64}),
            (DropoutLayer, {}),
            (DenseLayer, {'num_units': 64}),

            # the output layer
            (DenseLayer, {'num_units': 50, 'nonlinearity': tanh}),
         ]

    if net_num == 5:

        net_ret = [
            # layer dealing with the input data
            (InputLayer, {'shape': (None, 1, shape1, shape2)}),

           (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),
             (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),
             (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),

            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

            # first stage of our convolutional layers

             (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),
             (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),

            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

            # second stage of our convolutional layers
            (Conv2DLayerFast, {'num_filters': 64, 'filter_size': (3, 3)}),
            (Conv2DLayerFast, {'num_filters': 64, 'filter_size': (3, 3)}),

            #(NINLayer_c01b, {'num_units': 32}),
            (MaxPool2DLayerFast, {'pool_size':  (2, 2)}),

             # third stage of our convolutional layers
            (Conv2DLayerFast, {'num_filters': 128, 'filter_size': (3, 3)}),
                (Conv2DLayerFast, {'num_filters': 128, 'filter_size': (3, 3)}),

            #(NINLayer_c01b, {'num_units': 32}),
            (MaxPool2DLayerFast, {'pool_size':  (2, 2)}),



             # four stage of our convolutional layers
            (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),

            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

               # two dense layers with dropout
            (DenseLayer, {'num_units': 64, 'nonlinearity': None}),
            (DropoutLayer, {}),
            (DenseLayer, {'num_units': 64, 'nonlinearity': None}),


            # the output layer
            (DenseLayer, {'num_units': 50, 'nonlinearity': None}),
        ]

    if net_num == 6:

        net_ret = [
            # layer dealing with the input data
            (InputLayer, {'shape': (None, 1, shape1, shape2)}),

            # first stage of our convolutional layers
             (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),
            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

            # first stage of our convolutional layers
             (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),
            (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

            # second stage of our convolutional layers
            (Conv2DLayerFast, {'num_filters': 64, 'filter_size': (3, 3)}),
            #(NINLayer_c01b, {'num_units': 32}),
            (MaxPool2DLayerFast, {'pool_size':  (2, 2)}),

             # third stage of our convolutional layers
            (Conv2DLayerFast, {'num_filters': 128, 'filter_size': (3, 3)}),
            #(NINLayer_c01b, {'num_units': 32}),
            (MaxPool2DLayerFast, {'pool_size':  (2, 2)}),


            # two dense layers with dropout
            (DenseLayer, {'num_units': 128}),
            (DropoutLayer, {}),
            (DenseLayer, {'num_units': 128}),

            # the output layer
            (DenseLayer, {'num_units': 50, 'nonlinearity': tanh}),
        ]

    return net_ret

