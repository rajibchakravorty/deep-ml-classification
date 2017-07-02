from lasagne.layers import InputLayer, \
                           Conv2DLayer, \
                           MaxPool2DLayer, \
                           DenseLayer, \
                           dropout, \
                           ConcatLayer, \
                           BatchNormLayer, \
                           FlattenLayer, \
                           ReshapeLayer,\
                           FeaturePoolLayer, \
                           Pool2DLayer
from lasagne.nonlinearities import sigmoid, LeakyRectify, softmax
from lasagne.init import GlorotUniform



def cnn_archi( input_var, \
                  image_channel, image_height, image_width,\
                  output_length ):

    print image_height, image_width, image_channel

    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input_l = InputLayer(shape=(None, image_channel, \
                                      image_height, image_width) ,\
                                      input_var=input_var)

    # Input layer, as usual:
    network = Conv2DLayer( input_l, 128, (5,5) , pad = 'same' )
    network = MaxPool2DLayer( network, (2,2) )
    network = BatchNormLayer( network )
    
    network = Conv2DLayer( input_l,256, (3,3) ,pad = 'same' )
    network = MaxPool2DLayer( network, (2,2) )
    network = BatchNormLayer( network )

    network = Conv2DLayer( network,512 , (3,3) ,pad = 'same' )
    network = Conv2DLayer( network,512 , (3,3) ,pad = 'same' )
    network = Conv2DLayer( network,512 , (3,3) ,pad = 'same' )
    network = MaxPool2DLayer( network, (2,2) )
    network = BatchNormLayer( network )

 
    network = Conv2DLayer( 
                     network, 4096, (1,1), pad = 'same' )
    network = Conv2DLayer( 
                     network, 4096, (1,1), pad = 'same' )

    network = DenseLayer(
             network,
            num_units=output_length,
            nonlinearity=softmax)

    return network
