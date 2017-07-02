

import sys
sys.path.append( '../ml_utility' )

from image_object import ImageObject
from transformer import Transformer

from os import listdir
from os.path import isfile, join

import numpy as np

from random import random

image_dim = (26,26,3)


max_translation = 4
max_rotation    = 90.
do_flip         = True


num_label = 10


do_balance = True
augment_factor = 5

def prepare_original_imo_label( label_path, label ):

    image_files = [join(label_path, f ) for f in listdir( label_path ) \
                      if isfile( join( label_path, f ) ) ]

    imos = list()
    for imf in image_files:

        transform = Transformer( translation = (0,0 ),\
                                 rotation    = 0.0, \
                                 resize_shape = ( image_dim[0], image_dim[1] ),\
                                 do_flip = False,\
                                 convert_to_grey = False )

        imo = ImageObject( imf, label, transform )

        imos.append( imo )

    return imos

def prepare_original_imo( folder ):

    imos = list()

    count_labels = list()
    for label in range( num_label ):

        label_path = join( folder, '{0}'.format( label ) )
        imo_label  = prepare_original_imo_label( label_path, label )

        imos += imo_label

        count_labels.append( len( imo_label ) )

    return imos, count_labels

def balance_factor( count_labels ):

    count_labels = np.array( count_labels )

    factors = np.ones( count_labels.shape )

    max_idx = np.argsort( count_labels )

    max_count = count_labels[ max_idx[-1] ]

    factors = float( max_count ) /count_labels   

    return factors

def augment( factor, original_imos ):
 
    augmented_imos = list()   
    for imo in original_imos:

        for f in range( factor ):

            translation = ( int( ( random() - 0.5 ) * max_translation ),\
                        int( ( random() - 0.5 ) * max_translation ) )
            rotation = random() *max_rotation


            transform = Transformer( translation = translation,\
                                 rotation    = rotation, \
                                 resize_shape = ( image_dim[0], image_dim[1] ),\
                                 do_flip = do_flip,\
                                 convert_to_grey = False )

            aug_imo = ImageObject( imo.image_path , imo.label, transform )
 
            augmented_imos.append( aug_imo )

    return augmented_imos




def prepare_data( original_imos, count_labels, do_balance, aug_factor ):

    if do_balance == True:
        bfac = balance_factor( count_labels )

    additional_imos = list()
    start_idx = 0
    for l in range( num_label):

        end_idx = start_idx + count_labels[l]

        if do_balance == True:
            factor = int( bfac[l] * aug_factor )
        else:
            factor = aug_factor

        augmented_imos = augment( factor, original_imos[start_idx:end_idx] )

        start_idx = end_idx


        additional_imos += augmented_imos

    
    return additional_imos

def main():

    #train_folder = '/home/rajib/progs/MachineLearning/dataset/cifar-10-batches-py/train'
    #train_folder = '/home/ec2-user/miniconda2/envs/deeplearning/cifar/cifar-data/train'
    train_folder = '/home/rachakra/miniconda2/envs/deeplearning/cifar-demo/train'

    train_orig_imos, count_labels = prepare_original_imo( train_folder )
    np.random.shuffle( train_orig_imos )
    train_list = train_orig_imos[:-5000]
    valid_list = train_orig_imos[-5000:]

    train_aug_imos = prepare_data( train_list, count_labels, do_balance, augment_factor )
    print 'Train {0}, {1}'.format( len( train_orig_imos ), len( train_aug_imos ) )   
    train_list += train_aug_imos
    print 'Train {0}, {1}'.format( len( train_list ), len( valid_list ) )   
    #test_folder = '/home/ec2-user/miniconda2/envs/deeplearning/cifar/cifar-data/test'
    test_folder = '/home/rachakra/miniconda2/envs/deeplearning/cifar-demo/test'

    test_orig_imos, _ = prepare_original_imo( test_folder  )
    print 'Test {0}'.format( len( test_orig_imos ) )


    data_file = 'cifar-10.npz'

    np.savez( data_file, train_list, valid_list, test_orig_imos )

if __name__ == '__main__':

    main()    
