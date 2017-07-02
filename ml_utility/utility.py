import numpy as np

from random import random

from image_object import ImageObject


def save_epoch_info( epoch, epoch_time, train_loss, val_loss, val_accuracy, filename ):

    f = open( filename, 'a' )
    f.write('{0},{1},{2},{3},{4}\n'.format( epoch, epoch_time, \
                                             train_loss, \
                                             val_loss, \
                                             val_accuracy ) )
    f.close()



def iterate_minibatches( inputs, batchsize, shuffle=False):


    if shuffle:
        np.random.shuffle( inputs )

    #for start_idx in range(0, len(inputs), batchsize):
    for start_idx in range( 0, len( inputs), batchsize):
        end_idx = start_idx + batchsize

        if start_idx+batchsize >= len( inputs ):
            end_idx = len( inputs )

        yield inputs[start_idx:end_idx]




def iterate_minibatches2( inputs, batchsize, channel,\
                           height, width):


    #for start_idx in range(0, len(inputs), batchsize):
    for start_idx in range( 0, len( inputs), batchsize):
        end_idx = start_idx + batchsize

        if start_idx+batchsize >= len( inputs ):
            end_idx = len( inputs )

        X, y = imo2py( inputs[start_idx:end_idx], channel, \
                        height, width )

        #yield inputs[start_idx:end_idx]

        yield X,y

def imo2py( imo_list, channel, height, width ):

    list_length = len( imo_list )

    X = np.zeros( (list_length, channel, height, width ) )


    y = np.zeros( (list_length, ) )


    

    for idx, imos in enumerate( imo_list ):
        #print imos[0]


        im1 = imos.get_image()
        im1 = np.transpose( im1, (2,0,1) )

        X[idx, :, :, : ] = im1


        y[idx] = imos.label

    return X, y

