import json
import joblib
import numpy as np
import torch
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('image_resto')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    data = torch.from_numpy(data)/255
    data = data.view(-1, 224*224)
    
    # Get the reconstruction from the model
    reconstruction = model(data)

    # Return the predictions as JSON
    return json.dumps(reconstruction)


def scale_data(data):
    pass

def check_dims(data):
    pass