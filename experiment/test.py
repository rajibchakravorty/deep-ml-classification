from __future__ import print_function
import sys
import os
import time

import experiment

import numpy as np



from lasagne.layers import get_all_param_values

from sklearn.metrics import confusion_matrix, accuracy_score, \
                            classification_report, average_precision_score, \
                            roc_auc_score


from import_config import configuration as config


sys.path.insert( 0, '../ml_utility' )
import utility as ml_utility
from image_reader import ImageReader


num_readers = 30

def main():
   

    test_list = list( np.load( config.data_file )['arr_2'] )

    #print( type( train_list) ) #, type( train_aug_list ) )
    #print( len( train_list) ) # len( train_aug_list ) )
    print( 'Data detail' )
    print( '{0} - images for test'.format( \
                         len( test_list) ) )

    # Prepare Theano variables for inputs and targets
    test_fn =experiment.test_setup() 
    # Finally, launch the training loop.
    print("Starting testing")
    # We iterate over epochs:



    #print( train_batches )
    # And a full pass over the validation data:
    test_loss = 0.
    test_acc = 0.
    test_batches = 0
    print( 'validation starts ' )
    for batch in ml_utility.iterate_minibatches2( test_list, \
                                                     config.batch_size, \
                                                     config.image_channel, \
                                                     config.image_height, \
                                                     config.image_width ):
        #inputs, targets = ml_utility.imo2py( batch)
        inputs, targets = batch
        #assume color for now
        loss,pred_scores,acc = test_fn( inputs, targets )
        test_loss += np.sum( loss )
        test_acc += np.sum( acc )
        test_batches += 1

      

    # Then we print the results for this epoch:
    test_loss   = test_loss/ len( test_list ) 
    test_acc    = test_acc / len( test_list )



        
    #auc_score = roc_auc_score( val_truths, val_preds )
    print("  Test loss:\t\t{:.6f}".format( test_loss) )
    print("  Performance metrics " )
    print("  Test Accuracy : {0}".format( test_acc * 100. ) )
        #print("  AUC ROC:{0}".format( auc_score ) )
        



    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    

    main()

