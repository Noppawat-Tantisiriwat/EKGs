import tensorflow as tf
from tensorflow import keras
import warnings

from utils import *



class Pipeline:
    
    def __init__(self):
        
        self.cls = ['MI', 'Normal']

        

    def predict(self, image_path, model_path):

        try:
            self.image = batch(get_img_array(image_path))
            self.model = keras.models.load_model(model_path)

        except FileNotFoundError:
            
            warnings.warn("Image uploading failed")

        return get_prediction(self.image, self.model, self.cls)


        

if __name__ == "__main__":

    model_path = "mobilenet_plain_40_epochs.h5"

    pipeline = Pipeline()
    
    filename = "IMG_9054.heic"

    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><")
    pred = pipeline.predict(filename, model_path) 
    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><")
    

    print(f"Prediction of {filename} is {pred}.")






