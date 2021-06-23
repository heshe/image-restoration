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
from omegaconf import OmegaConf, DictConfig
from PIL import Image
from torch.optim import Adam
from pathlib import Path

import wandb
import pytorch_lightning as pl
import optuna
from src.models.model_lightning import ConvVAE, LoggingCallback
from src.models.model_FC import Decoder, Encoder, Net

log = logging.getLogger(__name__)



class Trainer:
    def __init__(self, args):
        self.args = args.experiment

        # Set root path
        self.ROOT = str(Path(__file__).parent.parent.parent)

    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(
            x_hat, x, reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss, KLD

    def loss_function2(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss, 0.01 * KLD

    def log_images_to_wandb(self, test_X, test_Y, model, img_size):
        # Generate reconstructions
        if self.args.use_wandb:

            n_images_to_log = 10  # Right now: cannot be more than #test_imgs = 1000

            model.eval()
            with torch.no_grad():

                X = test_X[:n_images_to_log, :, :]
                Y = test_Y[:n_images_to_log, :, :, :]

                Y = Y.permute(0, 3, 1, 2)
                X = resize(X, (img_size, img_size))
                Y = resize(Y, (img_size, img_size))

                if self.args.use_CNN:
                    X = X[:, None, :, :]

                else:
                    X = X.view(n_images_to_log, self.args.fc_flattened_dim)
                    Y = Y.view(n_images_to_log, self.args.fc_flattened_dim, 2)

                X = X.to(self.DEVICE)

                X_hat, _, _ = model(X)

            X_origin = get_rbg_from_lab(
                (X * 255).squeeze(),
                (Y * 255).permute(0, 2, 3, 1),
                img_size=img_size,
                n=n_images_to_log,
            )

            X_hat = get_rbg_from_lab(
                (X * 255).squeeze(),
                (X_hat * 255).permute(0, 2, 3, 1),
                img_size=img_size,
                n=n_images_to_log,
            )

            orig_images = []
            recon_images = []
            for i in range(n_images_to_log):
                im_o = Image.fromarray(X_origin[i])
                orig_images.append(im_o)
                im_r = Image.fromarray(X_hat[i])
                recon_images.append(im_r)

            wandb.log({"Originals": [wandb.Image(i) for i in orig_images]})

            wandb.log({"Reconstructed": [wandb.Image(i) for i in recon_images]})

    def train(self, trial=None):
        
        if self.args.optuna and trial:
            self.args.lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            self.args.latent_dim = trial.suggest_int("latent_dim", 32, 512)
            self.args.dropout = trial.suggest_uniform("dropout", 0.0, 1)

        img_size = (
            self.args.conv_img_dim
        )  # 33 if model_conv32, else 224 for model_conv224

        # Init wandb
        if self.args.use_wandb:
            wandb.init(config=self.args)

        # Training device
        self.DEVICE = torch.device(
            "cuda" if self.args.use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Get train and test
        if self.args.azure:
            from src.azure.make_dataset_azure import load_data
            from azureml.core import Run

            run = Run.get_context()  # Setup run instance for cloud
            
            datapath = run.input_datasets["image_resto"]
            run.log("datapath", datapath)
            run.log("args", self.args)

            train_dataloader = load_data(
                train=True,
                path=datapath,
                small_dataset=self.args.small_dataset,
                batch_size=self.args.batch_size,
            )
            test_dataloader = load_data(
                train=False,
                path=datapath,
                small_dataset=self.args.small_dataset,
                batch_size=self.args.batch_size,
            )

        else:
            from src.data.make_dataset import load_data

            train_dataloader = load_data(
                train=True,
                batch_size=self.args.batch_size,
                path = self.ROOT
            )
            test_dataloader = load_data(
                train=False,
                batch_size=self.args.batch_size,
                path = self.ROOT,
                shuffle=False,
            )

        # Init model
        model = ConvVAE(
            lr=self.args.lr,
            latent_dim=self.args.latent_dim,
            img_size=self.args.conv_img_dim,
            trial=trial
        )

        if self.args.azure:
            trainer = pl.Trainer(
                limit_train_batches=0.1, 
                max_epochs=self.args.n_epochs,
                precision=16,
                gpus=-1,
                callbacks=[LoggingCallback()]
            )
        else:
            trainer = pl.Trainer(
                #limit_train_batches=0.1, 
                max_epochs=self.args.n_epochs,
                callbacks=[LoggingCallback()]
            )


        #if self.args.use_wandb:
        #    wandb.watch(model, log_freq=100)

        print("Start training VAE...")
        trainer.fit(model, train_dataloader, test_dataloader)
        print("Finish training")
        """
        if self.args.use_wandb:
            self.log_images_to_wandb(X_test, Y_test, model, img_size)
        """

        # Save and register model in Azure
        if self.args.azure:
            if self.args.save_model:
                # Save the trained model
                model_file = config.experiment.model_name + ".pkl"
                tempmodel = ConvVAE() # Hack for saving model wihtout Pytorch Lightning things
                tempmodel.load_state_dict(model.state_dict())
                joblib.dump(value=tempmodel, filename=model_file)
                run.upload_file(
                    name=os.path.join(self.ROOT, "models", model_file),
                    path_or_stream="./" + model_file,
                )

                run.complete()
                # Register the model
                run.register_model(
                    model_path=os.path.join(self.ROOT, "models", model_file),
                    model_name=self.args.model_name,
                    tags={"Training context": "Inline Training"},
                )
            else:
                run.complete()
         
        return trainer.logged_metrics["val_loss"]


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


@hydra.main(config_path="../conf", config_name="config")
def init_hydra(config : DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    trainer = Trainer(config)
    if config.experiment.optuna:
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=5, 
                    interval_steps=1
            )
        ) #, sampler=optuna.samplers.GridSampler(search_space))
        
        study.optimize(trainer.train, n_trials=config.experiment.n_trials) 
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig2 = optuna.visualization.plot_intermediate_values(study)
        fig3 = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        )
        fig4 = optuna.visualization.plot_parallel_coordinate(study)
        fig1.write_image("opt_hist.jpg")
        fig2.write_image("lr_curves.jpg")
        fig3.write_image("param_importances.jpg")
        fig4.write_image("parallel_coordinates.jpg")
    else:
        trainer.train()

if __name__ == "__main__":
    config = init_hydra()
