import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import optuna
from kornia.geometry.transform import resize
from pytorch_lightning.callbacks import Callback

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
    def __init__(self, lr=0.001, img_size=33, latent_dim=256, dropout_rate=0.5, trial=None, run=None):
        super(ConvVAE, self).__init__()

        kernel_size = 3  # (4, 4) kernel
        init_channels = 64  # initial number of filters
        image_channels = 1  # MNIST images are grayscale
        self.latent_dim = latent_dim  # latent dimension for sampling
        self.lr = lr
        self.img_size = img_size
        self.dropout_rate = dropout_rate
        self.trial = trial
        self.run = run

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
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=0,
        )
        self.bn4 = nn.BatchNorm2d(init_channels*4)

        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 4,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.bn5 = nn.BatchNorm2d(init_channels*2)

        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 2,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.bn6 = nn.BatchNorm2d(init_channels)


        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels,
            out_channels=2,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.bn7 = nn.BatchNorm2d(2)


        self.dec5 = nn.ConvTranspose2d(
            in_channels=2, out_channels=2, kernel_size=kernel_size, stride=2, padding=1
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
        return val_loss

    def on_validation_epoch_end(self) -> None:
        if self.trial:
            self.trial.report(1, self.current_epoch)
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss_function2(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss, 0.01 * KLD

