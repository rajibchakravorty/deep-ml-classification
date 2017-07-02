import cPickle

from skimage.io import imsave

import numpy as np

import os

image_height = 32
image_width  = 32
image_channel = 3

max_label = 10

def save_label_image( all_labels, image_data, label, init_count = None ):

    select_labels = np.where( all_labels == label )
    select_labels = select_labels[0]
    
    if init_count is None:
        image_count = 0
    else:
        image_count = init_count
    for idx in range( select_labels.shape[0] ):
 
        l_idx = select_labels[idx]
        image_array = image_data[l_idx]

        image_2d = np.reshape( image_array, (image_channel, \
                                             image_height, \
                                             image_width ))

        image_2d = np.transpose( image_2d, (1,2,0 ) )

        filename = 'image_{0}.jpg'.format( image_count ) 
        image_path = os.path.join( '{0}'.format( label ), filename )

        imsave( image_path, image_2d )
        image_count += 1


    return image_count


def test():

   cifar_data_batches = ['data_batch_1',\
                         'data_batch_2',\
                         'data_batch_3',\
                         'data_batch_4',\
                         'data_batch_5',\
                         'test_batch']

   for current_label in range( max_label ):
       
       init_count = None

       for idx in range( len( cifar_data_batches ) - 1 ):
       #for idx in np.arange( start = 5, stop = 6):
           current_file = cifar_data_batches[idx]
           f = open( current_file, 'rb' )
           data_dict = cPickle.load( f )
           f.close()

           all_labels = np.array( data_dict['labels'] )


           image_data = data_dict[ 'data' ]

           init_count = save_label_image( all_labels, image_data, current_label, init_count )

           print init_count

if __name__ == '__main__':

   test()
