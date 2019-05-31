import json
import numpy as np
import os
# from keras.models import model_from_json
# from keras.layers import Input
from azureml.core.model import Model
from keras.models import Model as Model_keras
from squeezenet import SqueezeNet, preprocess_input


def init():

    global model

    model_root = Model.get_model_path('kd_teach_the_student') #, _workspace=ws)
    # model_root = model_root.strip('model.json')
    print(model_root)
    # load json and create model
    
    weight_file = os.path.join(model_root, "squeezenet_weights.hdf5")
    
    model = SqueezeNet(weight_decay=0.0, image_size=299, trainable=False, weight_file=weight_file)
    # model.load_weights(os.path.join(model_root, "squeezenet_weights.hdf5")) 

    model = Model_keras(model.input, model.outputs)
    
def run(raw_data):
    # convert json data to numpy array
    data = np.array(json.loads(raw_data)['data'])
    
    # make predictions
    X_hat = model.predict(data)

    return X_hat.tolist()
