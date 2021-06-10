# -*- coding: utf-8 -*-
import numpy as np
import logging
import torch
from pathlib import Path
import matplotlib.pyplot as plt

color_path = "C:/Users/Bruger/Desktop/DTU/MLOps/Kaggle_data/l/"



def load_MIRFLICKR25k():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    #data = torch.load(color_path + "ab2.npy")
    data = np.load(color_path + "gray_scale.npy")
    data_torch = torch.from_numpy(data)
    print(data_torch.size())
    
    print(data_torch[0].size())

    data[0].shape
    #view = data[0].view(224,224,2)
    plt.imshow(data[0].astype("float32"))
    plt.show()
    plt.plot(np.array([1,2,3,4]),np.array([1,2,3,4]))

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_MIRFLICKR25k()
