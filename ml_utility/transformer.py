
from skimage.color import rgb2gray
from skimage.transform import resize,\
                              rotate
import numpy as np
from random import random


class Transformer():


    def __init__( self, translation = (0,0),\
                        rotation = 0.0,\
                        resize_shape = None, \
                        do_flip = False, \
                        convert_to_grey = False ):

        # translation is assumed to be tuple of length 2
        # assumed to be integer; NO ERROR CHECK

        # rotation is assumed to be in degrees

        self.convert_to_grey = convert_to_grey
        self.resize_shape    = resize_shape
        self.rotation        = rotation
        self.do_flip         = do_flip

        self.translation     = translation

    def process_image( self, im ):
        
        if self.convert_to_grey == True:

            im =  rgb2gray( im )

        im = self._image_dim_convert( im )

        if not ( self.resize_shape is None) :

            im = resize( im, self.resize_shape )

        if np.max( im ) > 1.0:
            im = im / 255.

        im = self._flip_image( im )
        im = self._translate_image( im )
        im = self._rotate_image( im )


        return im

    def _get_image_dim( self, im ):

        image_height = im.shape[0]
        image_width  = im.shape[1]
        image_channel = 1
        if( len( im.shape) > 2 ):
            image_channel = im.shape[2]

        return image_height, image_width, image_channel

    def _image_dim_convert( self, im ):

        h,w,c = self._get_image_dim( im )

        return np.reshape( im, (h,w,c) )

    def _translate_image( self, im ):

        sum_translation = self.translation[0] + self.translation[1]

        if( sum_translation ==0 ):
            return im

        h,w,c = self._get_image_dim( im )

        

        copy_image = np.zeros( ( h + 2*np.abs( self.translation[0] ),\
                                 w + 2*np.abs( self.translation[1] ),\
                                 c ) )

        copy_image[ np.abs( self.translation[0]):( np.abs(self.translation[0] ) + h) ,
                    np.abs( self.translation[1]):( np.abs(self.translation[1] ) + w), :] = \
                     im

        top_y = np.abs( self.translation[0] ) + self.translation[0]

        top_x = np.abs( self.translation[1] ) + self.translation[1]

        translated_image = copy_image[ top_y:( top_y + h ), top_x:(top_x + w), : ]

        return translated_image 

    def _rotate_image( self, im ):

        if self.rotation == 0.0:
            return im

        else:

            return rotate( im, self.rotation, resize = False )


    def _flip_image( self, im ):

        if self.do_flip== True:
        
           rand = random()

           _,_,c = self._get_image_dim( im )

           copy_image = im
           for ch in range( c) :
               if rand > 0.5:

                   flipped_channel = np.fliplr( im[:,:,ch ] )

                   copy_image[:,:,ch] = flipped_channel

               elif rand < 0.5:

                   flipped_channel = np.flipud( im[:,:,ch ] )

                   copy_image[:,:,ch] = flipped_channel


           return copy_image

        else:

           return im


if __name__ == '__main__':

    from skimage.io import imread, imsave

    im = imread( '/home/rajib/progs/MachineLearning/dataset/cifar-10-batches-py/train/3/image_22.jpg' )

    transformer = Transformer( translation = (0,0),\
                               rotation = 0.,\
                               resize_shape = (26,26), \
                               do_flip = False, \
                               convert_to_grey = False )


    im = transformer.process_image( im )

    #im = np.reshape( im, (im.shape[0], im.shape[1] ) )  
    imsave( 'test.jpg', im )
