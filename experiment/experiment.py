

import theano
import theano.tensor as T
import lasagne

from lasagne.regularization import regularize_network_params, l1, l2
from lasagne.updates import nesterov_momentum, rmsprop, \
                            norm_constraint, adagrad, \
                            total_norm_constraint

from cnn_1 import cnn_archi

from lasagne.layers import get_all_params, get_all_param_values, \
                           set_all_param_values, count_params, \
                           get_output
from lasagne.objectives import categorical_crossentropy
from lasagne.utils import compute_norms
import numpy as np

from import_config import configuration as config




def test_setup():


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    print( " with input dimension {0},{1},{2}".format( config.image_height, \
                                                       config. image_width, \
                                                       config.image_channel ) )
    network = cnn_archi( input_var,  \
                         config.image_channel, \
                         config.image_height, config.image_width,\
                         config.output_length )

    print( 'Number of parameters : {0}'.format( count_params( network ) ) )

    with np.load(config.model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    set_all_param_values(network, param_values)
    
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_classes    = T.argmax( test_prediction, axis = 1 )
    test_loss       = categorical_crossentropy(test_prediction,\
                                                            target_var)
    
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.eq( test_classes, target_var)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var,  target_var], \
                             [test_loss, test_prediction, test_acc], \
                             allow_input_downcast=True )

    return val_fn



def train_setup():


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    print( " with input dimension {0},{1},{2}".format( config.image_height, \
                                                       config.image_width, \
                                                      config. image_channel ) )
    network = cnn_archi( input_var,  \
                         config.image_channel,\
                         config.image_height, config.image_width,\
                         config.output_length )

    print( 'Number of parameters : {0}'.format( count_params( network ) ) )

    if (config.init_model is not None):
        with np.load(config.init_model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        set_all_param_values(network, param_values)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    ent_loss = categorical_crossentropy(prediction, target_var)
    ent_loss = ent_loss.mean()
    
    l1_regu = config.l1_regu * regularize_network_params(network, l1)
    l2_regu = config.l2_regu * regularize_network_params(network, l2)

    loss = ent_loss + l1_regu + l2_regu
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = get_all_params(network, trainable=True)

    #grads = T.grad( loss, params )
    #scaled_grads = norm_constraint( grads, 5. )
    updates = nesterov_momentum(loss, params, \
                                learning_rate=config.learning_rate, \
                                momentum=config.momentum )
    #updates = rmsprop( loss , params, learning_rate = config.learning_rate )


    for param in get_all_params( network, regularizable = True ):
        norm_axis = None
        if param.ndim == 1:
            norm_axis = [0]
        updates[param] = norm_constraint( updates[param], \
                                 5. * compute_norms( param.get_value() ).mean(),
                                 norm_axes = norm_axis  )


    #Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = get_output(network, deterministic=True)
    test_classes    = T.argmax( test_prediction, axis = 1 )
    test_loss       = categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.eq(test_classes, target_var)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var,target_var], \
                               ent_loss,\
                               updates=updates, \
                               allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], \
                             [test_loss, test_prediction, test_acc], \
                             allow_input_downcast=True )

    return network, train_fn, val_fn
