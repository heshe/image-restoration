import json
import os

import joblib
import cv2
import numpy as np
import torch
from pathlib import Path
from kornia.geometry.transform import resize


# Called when the service is loaded
def init():
    #project_dir = str(Path(__file__).parent.parent.parent)
    global model
    # Get the path to the deployed model file and load it
    print(os.getenv("AZUREML_MODEL_DIR"))
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "image_resto.pkl")
    model = joblib.load(model_path)


# Called when a request is received
def run(data):
    # Get the input data as a numpy array
    X = np.array(json.loads(data)["input_data"])
    X = torch.from_numpy(X)
    X = resize(X, (224, 224))

    X = X[:, None, :, :]
    # Get the reconstruction from the model
    X_hat, _, _ = model(X.float())
    X_hat = X_hat.detach()

    n = X_hat.shape[0]

    X_hat = get_rbg_from_lab(
        (X * 255).squeeze(), 
        (X_hat * 255).permute(0, 2, 3, 1), 
        224, 
        n=n
    )

    # Return the predictions as JSON
    return json.dumps(X_hat.tolist())


def get_rbg_from_lab(gray_imgs, ab_imgs, img_size, n=10):
    # create an empty array to store images
    imgs = np.zeros((n, img_size, img_size, 3))

    imgs[:, :, :, 0] = gray_imgs[0:n:]
    imgs[:, :, :, 1:] = ab_imgs[0:n:]

    # convert all the images to type unit8
    imgs = imgs.astype("uint8")

    # create a new empty array
    imgs_ = []

    for i in range(0, n):
        imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))

    # convert the image matrix into a numpy array
    imgs_ = np.array(imgs_)

    return imgs_