import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from PIL import Image

try:
    import pyheif
except:
    pass


def get_img_array(img_path):
    """
    Apple device might have .HEIC as outputs. The OS must be UNIX to run this. 
    pyheif must be installed.
    """
    if img_path.split('.')[-1] == 'heic':
    
        heif_file = pyheif.read(img_path)

            
        im = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )


        image = np.asarray(im.resize((550, 375))).astype('float32')

        return image
    
    elif img_path.split('.')[-1] == 'png':
        
        im = Image.open(img_path)

        image = np.array(im.resize((550, 375)))[:,:,:3]


        return image


    elif img_path.split('.')[-1] == 'jpg':

        im = Image.open(img_path)

        image = np.array(im.resize((550, 375)))


        return image


    else:
        try:
            im = Image.open(img_path)

            image = np.array(im.resize((550, 375)))


            return image
        
        except:
            print("Unable to load the image.")
        



def batch(data : np.ndarray):

    return np.expand_dims(data, axis=0)


def get_prediction(data, model, cls :List):    
    """
    Get the prediction from multiclass classifier. Data must be batched
    """
    assert type(data) == np.ndarray
    pred = model.predict(data)[0]
    prediction = tf.argmax(pred)

    return cls[prediction]
