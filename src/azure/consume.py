import urllib.request
import json
import os
import ssl
from cv2 import data
import numpy as np
from pathlib import Path

import torch
from kornia.geometry.transform import resize
from src.models.model_lightning import ConvVAE
from src.data.make_dataset import load_data
import cv2
import matplotlib.pyplot as plt
from skimage import color

"""
def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

#// Request data goes here

datapath = str(Path(__file__).parent.parent.parent) + "/data/interim/exam_images.npy"
X = np.load(datapath)

for i, img in enumerate(X, 1):
    plt.figure(i)
    plt.imshow((img).squeeze(), cmap="gray")
plt.show()

choice = input("Choose an input image\n")

X = X[int(choice)-1]
X = X[np.newaxis, ...]
X = X.tolist()

data = {"input_data" : X}
body = str.encode(json.dumps(data))

url = 'http://7e21269b-a087-4a7f-88c9-07b5a94cff7e.northeurope.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
    print(type(result))
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))
"""

def run(data):
    # Get the input data as a numpy array
    X = np.array(json.loads(data)["input_data"])
    X = torch.from_numpy(X)
    X = resize(X, (224, 224))

    X = X[:, None, :, :]
    model = ConvVAE()
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

    #for i in range(0, n):
    #    imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))

    # convert the image matrix into a numpy array
    imgs_ = np.array(imgs)

    return imgs_





if __name__ == "__main__":

    datapath = str(Path(__file__).parent.parent.parent) + "/data/interim/exam_images.npy"
    X = np.load(datapath)

    for i, img in enumerate(X, 1):
        plt.figure(i)
        plt.imshow((img).squeeze(), cmap="gray")
    plt.show()

    choice = input("Choose an input image\n")

    X = X[int(choice)-1]
    X = X[np.newaxis, ...]
    X = X.tolist()

    data = {"input_data" : X}
    body = str.encode(json.dumps(data))
    img = np.array(json.loads(run(body))).astype(np.uint8)
    plt.figure()
    plt.imshow((img*255).squeeze())
    plt.show()

    


