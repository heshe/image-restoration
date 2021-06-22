
"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import logging

import cv2
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import tqdm
import joblib
from kornia.geometry.transform import resize
from omegaconf import OmegaConf
from PIL import Image
from torch.optim import Adam
from pathlib import Path

import wandb
#from src.models.model_conv32 import ConvVAE
from src.models.model_lightning import ConvVAE
from src.models.model_FC import Decoder, Encoder, Net
from azureml.core import Run

import pytorch_lightning as pl

log = logging.getLogger(__name__)

ROOT = str(Path(__file__).parent.parent.parent)

hydra.initialize(config_path="../conf")
config = hydra.compose(config_name="config.yaml")
print(f"configuration: \n {OmegaConf.to_yaml(config)}")

if config.experiment.azure:
    from src.azure.make_dataset_azure import load_data

    run = Run.get_context()  # Setup run instance for cloud
    datapath = run.input_datasets["image_resto"]
    run.log("datapath", datapath)
    run.log("args", config.experiment)

    train_dataloader = load_data(
        train=True,
        path=datapath,
        small_dataset=config.experiment.small_dataset,
        batch_size=config.experiment.batch_size,
    )
    test_dataloader = load_data(
        train=False,
        path=datapath,
        small_dataset=config.experiment.small_dataset,
        batch_size=config.experiment.batch_size,
    )

else:
    from src.data.make_dataset import load_data

    train_dataloader = load_data(
        train=True,
        batch_size=config.experiment.batch_size,
    )
    test_dataloader = load_data(
        train=False,
        batch_size=config.experiment.batch_size,
    )

model = ConvVAE(
    lr=config.experiment.lr,
    img_size=config.experiment.conv_img_dim
)

trainer = pl.Trainer(
    limit_train_batches=0.1, 
    max_epochs=config.experiment.n_epochs,
    precision=16,
    gpus=-1,
)

trainer.fit(model, train_dataloader, test_dataloader)

print("Training Finished!")

if config.experiment.azure:
    if config.experiment.save_model:
        # Save the trained model
        model_file = config.experiment.model_name + ".pkl"
        tempmodel = ConvVAE() # Hack for saving model wihtout Pytorch Lightning things
        tempmodel.load_state_dict(model.state_dict())
        joblib.dump(value=tempmodel, filename=model_file, protocol=4)
        run.upload_file(
            name=os.path.join(ROOT, "models", model_file),
            path_or_stream="./" + model_file,
        )
        run.complete()
        # Register the model
        run.register_model(
            model_path=os.path.join(ROOT, "models", model_file),
            model_name=config.experiment.model_name,
            tags={"Training context": "Inline Training"},
            #properties={
            #    "LR": run.get_metrics()["LR"],
            #    "Epochs": run.get_metrics()["Epochs"],
            #    "Latent dim": run.get_metrics()["Latent dim"],
            #    "Hidden dim": run.get_metrics()["Hidden dim"],
            #    "Overall loss": run.get_metrics()["Overall loss"],
            #},
        )
    else:
        run.complete()
