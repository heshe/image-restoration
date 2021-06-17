import json
import os

import dotenv
import joblib
import numpy as np
import torch
from azureml.core.model import Model


# Called when the service is loaded
def init():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, ".env")
    dotenv.load_dotenv(dotenv_path)
    global model
    # Get the path to the deployed model file and load it
    print(os.getenv("AZUREML_MODEL_DIR"))
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "image_resto.pkl")
    model = joblib.load(model_path)


# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)["data"])
    data = torch.from_numpy(data) / 255
    data = data.view(-1, 224 * 224)

    # Get the reconstruction from the model
    reconstruction = model(data.float())

    # Return the predictions as JSON
    return json.dumps([rec.tolist() for rec in reconstruction])


def scale_data(data):
    pass


def check_dims(data):
    pass
