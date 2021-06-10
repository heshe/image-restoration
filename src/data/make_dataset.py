# -*- coding: utf-8 -*-
import numpy as np
import logging
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from PIL import Image


color_path = "/Users/heshe/Desktop/mlops/image-restoration/data/raw/"



def load_MIRFLICKR25k():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    #data = torch.load(color_path + "ab2.npy")
    ab_ab_imgs = np.load(color_path + "/ab/ab/ab1.npy")
    gray_imgs = np.load(color_path + "/l/gray_scale.npy")
    data_torch = torch.from_numpy(data)
    print(data_torch.size())
    
    print(data_torch[0].size())

    data[0].shape
    #view = data[0].view(224,224,2)
    rgb_image = cv2.cvtColor(data[0].astype(np.uint8), cv2.COLOR_LAB2RGB)

    rgb = cv2.cvtColor(data[0], cv2.COLOR_LAB2RGB)

    plt.imshow(data[0])
    plt.imshow(rgb_image)
    plt.show()
    plt.plot(np.array([1,2,3,4]),np.array([1,2,3,4]))

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


def get_rbg_from_lab(gray_imgs, ab_imgs, n = 10):
    
    #create an empty array to store images
    imgs = np.zeros((n, 224, 224, 3))
    rgb_imgs = np.zeros((n, 224, 224, 3)).astype("uint8")
    
    imgs[:, :, :, 0] = gray_imgs[0:n:]
    imgs[:, :, :, 1:] = ab_imgs[0:n:]
    
    #convert all the images to type unit8
    imgs = imgs.astype("uint8")
    
    for i in range(0, n):
        A = cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB).astype("uint8")
        im = Image.fromarray(A)
        im.save(f"/Users/heshe/Desktop/mlops/image-restoration/data/img{i}.png")

    #convert the image matrix into a numpy array
    imgs_ = np.array(imgs_)

    #print(imgs_.shape)
    
    return imgs_
    
#preprocess the input to 
imgs_for_output = preprocess_input(get_rbg_from_lab(gray_imgs = images_gray, ab_imgs = images_lab, n = 300))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_MIRFLICKR25k()
