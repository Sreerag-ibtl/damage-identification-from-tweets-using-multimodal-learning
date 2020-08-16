from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import tensorflow_hub as hub
import re
import numpy as np

class Prediction:
    """A class that implements prediction."""
    
    def __init__( self, text_model_path, image_model_path  ):

        self.text_model  = load_model( text_model_path, custom_objects={"KerasLayer":hub.KerasLayer} )
        self.image_model = load_model( image_model_path )

    def run( self, image, text ):

        image = self.preprocess_image( image )
        text  = self.preprocess_text( text ) 

        im_preds  = self.image_model.predict( image )
        im_class  = np.argmax( im_preds )
        txt_preds = self.text_model.predict( text )
        txt_class = np.argmax( txt_preds )

        return im_class, txt_class

    def preprocess_image( self, image ):

        image = cv2.resize( image, ( 299, 299 ) )
        image = image / 255.
        image = np.expand_dims( image, axis = 0 )
        return image

    def preprocess_text( self, text ):

        text = re.sub( "[#=_()~:,@./+!?|]", "", text )
        text = re.sub( "[^A-Za-z0-9\x00-\x7F]+", "", text)
        text = text.replace( "\n", "" )
        text = text.lower( )
        text = text.encode()
        text = np.array( text )
        text = np.expand_dims( text, 0 )
        text = tf.convert_to_tensor( text )
        return text

if __name__ == "__main__":

    path  = "../test_images/damaged_building.jpg"
    
    image = cv2.imread( path )
    text  = "happy birthday"

    prediction = Prediction( "../models/text.h5", "../models/old_image/damage.h5" )
    prediction.run( image, text )
    
