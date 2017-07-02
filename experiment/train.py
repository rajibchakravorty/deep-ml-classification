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

from Queue import Queue
from threading import Thread

sys.path.insert( 0, '../ml_utility' )
import utility as ml_utility
from image_reader import ImageReader




source_queue = Queue()
dest_queue   = Queue()

num_readers = 30

def main():
   

    train_list = list( np.load( config.data_file )['arr_0'] )
    valid_list = list ( np.load( config.data_file)['arr_1'] )

    #print( type( train_list) ) #, type( train_aug_list ) )
    #print( len( train_list) ) # len( train_aug_list ) )



    #train_list = train_list[0:150000]
    #valid_list = valid_list[0:1000]
    print( 'Data detail' )
    print( '{0}/{1} - images for training/validation'.format( \
                         len( train_list) , len( valid_list) ) )

    # Prepare Theano variables for inputs and targets
    net, train_fn, val_fn = experiment.train_setup() 
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:


    reader_threads = []
    for th in range( num_readers  ):
        t = ImageReader( dest_queue, source_queue, \
                        config.image_height, config.image_width, \
                         config.image_channel )
        t.setDaemon( True )
        t.start()
        #reader_threads.append ( t )
    #for th in reader_threads:
    #    th,start()


    start_loss = config.start_loss
    for epoch in range( config.num_epochs):
        

        print( 'Running epoch: {0}'.format( epoch+1 ) )
        # In each epoch, we do a full pass over the training data:
        train_loss = 0.
        train_batches = 0
        start_time = time.time()
        


        for batch in ml_utility.iterate_minibatches( train_list, \
                                                    config.batch_size, \
                                                    shuffle=True):

            source_queue.put( batch )
            train_batches += 1
            #print( 'Loading {0} batches'.format( train_batches ) )

        train_batches -= 1      


        source_counter = source_queue.qsize()
        dest_counter   = 0
        while dest_counter < train_batches:
            if dest_queue.qsize() > 0 :
                
                #print( 'Started batch training' )
                inputs, targets = dest_queue.get()
                dest_queue.task_done()

                #assume color for now
                loss = train_fn( inputs, targets )
                train_loss += loss
                dest_counter += 1

                print( 'trained with {0} batches'.format( dest_counter ) )

        
        

        #print( train_batches )
        # And a full pass over the validation data:
        val_loss = 0.
        val_acc = 0.
        val_batches = 0
        val_preds = None
        val_truths = None
        print( 'validation starts ' )
        for batch in ml_utility.iterate_minibatches2( valid_list, \
                                                     config.batch_size, \
                                                     config.image_channel, \
                                                     config.image_height, \
                                                     config.image_width ):
            #inputs, targets = ml_utility.imo2py( batch)
            inputs, targets = batch
            #assume color for now
            loss,pred_scores,acc = val_fn( inputs, targets )
            pred_scores = pred_scores[:,1]
            if val_preds is None:
                val_preds = pred_scores
                val_truths = targets
            else:
                val_preds = np.concatenate( ( val_preds, pred_scores), \
                                            axis = 0 )
                val_truths = np.concatenate( (val_truths, targets ), \
                                            axis = 0 )  
            val_loss += loss
            val_acc += np.sum( acc )
            val_batches += 1

      

        # Then we print the results for this epoch:
        epoch_time = time.time() - start_time
        train_loss = train_loss/ train_batches
        val_loss   = val_loss/ val_batches 
        val_acc    = val_acc / len( valid_list )


        val_preds=np.array( val_preds)
        val_truths = np.array( val_truths )

        val_preds = np.reshape( val_preds, (-1,1 ) ) 
        val_truths = np.reshape( val_truths, (-1,1 ) )
        val_class  = np.argmax( val_preds, axis = 1 )
        
        #auc_score = roc_auc_score( val_truths, val_preds )
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, config.num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_loss) )
        print("  validation loss:\t\t{:.6f}".format( val_loss) )
        print("  Performance metrics " )
        print("  Validation Accuracy : {0}".format( val_acc * 100. ) )
        #print("  AUC ROC:{0}".format( auc_score ) )
        

        ml_utility.save_epoch_info( epoch+1,\
                                    epoch_time,\
                                    train_loss, \
                                    val_loss, \
                                    val_acc, \
                                    config.stat_file )


        #save the model after every 10 iterations
        if( val_loss <= start_loss ):

            np.savez( config.model_file, *get_all_param_values(net) )
            start_loss = val_loss


    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    

    main()

