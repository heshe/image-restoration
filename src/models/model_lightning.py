import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import optuna
import cv2
import os
from kornia.geometry.transform import resize
from pytorch_lightning.callbacks import Callback
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

class LoggingCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.logged_metrics["val_loss"]
        epoch = trainer.logged_metrics["epoch"]
        if pl_module.trial:
            pl_module.trial.report(val_loss.item(), int(epoch.item()))
            
            if pl_module.trial.should_prune():
                raise optuna.TrialPruned()
        
        if pl_module.run:
            pl_module.run.log("Val loss", trainer.logged_metrics["val_loss"].item())
        
    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.run:
            pl_module.run.log("Train loss", trainer.logged_metrics["train_loss"].item())

class ConvVAE(pl.LightningModule):
    def __init__(self, lr=0.001, img_size=33, latent_dim=1024, dropout_rate=0.5, trial=None, run=None):
        super(ConvVAE, self).__init__()

        kernel_size = 4  # (4, 4) kernel
        init_channels = 80  # initial number of filters
        image_channels = 1  # MNIST images are grayscale
        self.latent_dim = latent_dim  # latent dimension for sampling
        self.lr = lr
        self.img_size = img_size
        self.dropout_rate = dropout_rate
        self.trial = trial

        # Logging and reporting
        self.run = run
        self.first_run = True
        self.ROOT = str(Path(__file__).parent.parent.parent)
        self.round = 0 # Used to avoid name conflict in files


        # ____________________ENCODER____________________
        self.enc1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(init_channels)

        self.enc2 = nn.Conv2d(
            in_channels=init_channels,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(init_channels*2)

        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 2,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(init_channels*4)


        # fully connected layers for learning representations
        self.fc1 = nn.Linear(init_channels * 4, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

        # ____________________DECODER____________________
        self.dec1 = nn.ConvTranspose2d(
            in_channels=self.latent_dim,
            out_channels=init_channels * 16,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        self.bn4 = nn.BatchNorm2d(init_channels*16)

        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 16,
            out_channels=init_channels * 8,
            kernel_size=kernel_size,
            stride=2,
            padding=2,
        )
        self.bn5 = nn.BatchNorm2d(init_channels*8)

        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 8,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=3,
            padding=2,
        )
        self.bn6 = nn.BatchNorm2d(init_channels*4)


        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*4,
            out_channels=init_channels * 2,
            kernel_size=kernel_size +1,
            stride=4,
            padding=2,
        )
        self.bn7 = nn.BatchNorm2d(init_channels*2)

        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels*2, 
            out_channels=2, 
            kernel_size=kernel_size + 2, 
            stride=4, 
            padding=3
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):
        dropout_rate = self.dropout_rate
        # ____________________ENCODING____________________
        x = self.bn1(F.dropout(F.relu(self.enc1(x)), dropout_rate))
        x = self.bn2(F.dropout(F.relu(self.enc2(x)), dropout_rate))
        x = self.bn3(F.dropout(F.relu(self.enc3(x)), dropout_rate))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)

        # ____________________REPARAMETERIZATION____________________
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, self.latent_dim, 1, 1)

        # ____________________DECODING____________________
        x = self.bn4(F.relu(self.dec1(z)))
        x = self.bn5(F.relu(self.dec2(x)))
        x = self.bn6(F.relu(self.dec3(x)))
        x = self.bn7(F.relu(self.dec4(x)))
        reconstruction = torch.relu(self.dec5(x))
        return reconstruction, mu, log_var

    def training_step(self, batch, batch_idx):
        X, Y = batch   
        Y = Y.permute(0, 3, 1, 2)

        X = resize(X, (self.img_size, self.img_size))
        Y = resize(Y, (self.img_size, self.img_size))

        X = X[:, None, :, :]

        X_hat, mean, log_var = self(X)
        rec, kld = self.loss_function2(Y, X_hat, mean, log_var)
        loss = rec + kld
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch   
        Y = Y.permute(0, 3, 1, 2)
        X = resize(X, (self.img_size, self.img_size))
        Y = resize(Y, (self.img_size, self.img_size))

        X = X[:, None, :, :]

        X_hat, mean, log_var = self(X)
        rec, kld = self.loss_function2(Y, X_hat, mean, log_var)
        val_loss = rec + kld
        self.log('val_loss', val_loss)

        if batch_idx==0:
            if self.run:
                X_cpu = X.to('cpu')
                X_hat_cpu = X_hat.to('cpu')
                Y_cpu = Y.to('cpu')
                self.log_images_to_azure(X_cpu, Y_cpu, X_hat_cpu)
                self.first_run = False

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss_function2(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss, 0.01 * KLD

    def get_rbg_from_lab(self, gray_imgs, ab_imgs, img_size, n=10):
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


    def log_images_to_azure(self, X, Y, X_hat):
        n_images_to_log = 4 # Right now: cannot be more than #test_imgs = 1000

        X = X[:n_images_to_log, :, :, :]
        X_hat = X[:n_images_to_log, :, :, :]
        Y = Y[:n_images_to_log, :, :, :]

        X_origin = self.get_rbg_from_lab(
            (X * 255).squeeze(),
            (Y * 255).permute(0, 2, 3, 1),
            img_size=self.img_size,
            n=n_images_to_log,
        )

        X_hat = self.get_rbg_from_lab(
            (X * 255).squeeze(),
            (X_hat * 255).permute(0, 2, 3, 1),
            img_size=self.img_size,
            n=n_images_to_log,
        )

        if self.first_run: # Log originals in first run
            for i, img in enumerate(X_origin):
                img_path = os.path.join(self.ROOT, "reports", "figures", "orig", f"orig{i}.jpg")
                im_o = Image.fromarray(img)
                plt.figure()
                plt.imshow(im_o)
                plt.savefig(img_path)
                self.run.log_image(
                    name=f"orig{i}",
                    path=img_path
                )

            
        for i, img in enumerate(X_hat):
            img_path = os.path.join(self.ROOT, "reports", "figures", "recon", f"recon{i}.jpg")
            im_r = Image.fromarray(img)
            plt.figure()
            plt.imshow(im_r)
            plt.savefig(img_path)
            self.run.log_image(
                name=f"recon{i}_{self.round}",
                path=img_path
            )
        self.round += 1


        

