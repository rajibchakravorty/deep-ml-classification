import numpy as np

from image_object import ImageObject

from Queue import Queue, Empty
from threading import Thread


class ImageReader( Thread ):


    def __init__( self, dest_queue, source_queue, \
                  image_height, image_width, image_channel ):

        Thread.__init__( self )
        self.dest_queue = dest_queue
        self.source_queue = source_queue

        self.channel    = image_channel
        self.image_size = (image_height, image_width )


    def run( self ):

        while True:
            X,y = self.imo2py(  )
            if X is not None:
                self.dest_queue.put( (X,y) )

          

    def imo2py( self ):

        #if self.source_queue.empty():
        #    return None, None

        try:
            imo_list = self.source_queue.get()
            list_length = len( imo_list )

            X = np.zeros( (list_length, self.channel, \
                                  self.image_size[0],\
                                  self.image_size[1] ) )


            y = np.zeros( (list_length,  ) )



            for idx, imos in enumerate( imo_list ):

            
                #print imos[0]
                im1 = imos.get_image()
                im1 = np.transpose( im1, (2,0,1) )
                X[idx, :, :, : ] = im1


                y[idx] = imos.label


            self.source_queue.task_done()
            return X, y

        except Empty:
            return None, None    

