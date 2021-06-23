import urllib.request
import json
import os
import ssl
from cv2 import data
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


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

X = X.tolist()

data = {"input_data" : X}
body = str.encode(json.dumps(data))

url = 'http://23834561-8e02-4766-ad77-de95f45f0dd3.northeurope.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)
    
    result = response.read().decode('utf-8')
    imgs = np.array(json.loads(json.loads(result))).astype(np.uint8)
    for i, img in enumerate(imgs, 1):
        plt.figure(i)
        plt.imshow((img).squeeze())
    plt.show()


except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))