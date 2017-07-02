from skimage.io import imread
from random import random


from transformer import Transformer as tformer

class ImageObject():


    def __init__( self, image_path, label, \
                  transformer ):

        self.image_path = image_path
        self.label = label
        self.transformer = transformer

    def get_image( self ):
        im = imread( self.image_path )

        return self.transformer.process_image( im )


def test():

    image_path = '/home/rajib/progs/MachineLearning/dataset/cifar-10-batches-py/train/3/image_22.jpg'

    label = 3

    max_rotation = 0.
    max_translation = 0
    
    translation = ( int( ( random() - 0.5 ) * max_translation ),\
                    int( ( random() - 0.5 ) * max_translation ) )
    rotation = random() *max_rotation
    resize_shape = (26,26)
    do_flip = False
    to_grey = False
    print 'Transformation...'
    print 'Translation: {0}'.format( translation )
    print 'Rotation: {0}'.format( rotation )
    print 'Flip : {0}'.format( do_flip )
    print 'To Grey? {0}'.format( to_grey )

    t = tformer( translation = translation, \
                 rotation = rotation, \
                 resize_shape = resize_shape, \
                 do_flip = do_flip, \
                 convert_to_grey = to_grey )
    imo = ImageObject( image_path, label, t )

    im = imo.get_image()
    print im.shape
    from skimage.io import imsave
    import numpy as np
    imsave( 'test.jpg', im )

if __name__ == '__main__':

    test()
