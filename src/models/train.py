"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import argparse
import sys

# sys.path.insert(0,"C:/Users/Asger/OneDrive/Skrivebord/DTU/Machine_Learning_Operations/image-restoration")

import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
from kornia.geometry.transform import resize
from PIL import Image
from torch.optim import Adam

import wandb
from src.models.model_conv32 import ConvVAE
from src.models.model_FC import Decoder, Encoder, Net
from src.data.make_dataset import load_data


class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--n_epochs", default=5, type=int)
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--fc_flattened_dim", default=224 * 224, type=int)
        parser.add_argument("--fc_latent_dim", default=20, type=int)
        parser.add_argument("--fc_hidden_dim", default=400, type=int)
        parser.add_argument("--conv_img_dim", default=33, type=int)

        parser.add_argument("--use_wandb", default=True, type=bool)
        parser.add_argument("--plot_results", default=True, type=bool)
        parser.add_argument("--use_cuda", default=False, type=bool)
        parser.add_argument("--use_CNN", default=True, type=bool)

        parser.add_argument(
            "--dataset_path",
            default="C:/Users/Asger/OneDrive/Skrivebord/DTU/Machine_Learning_Operations/data",
        )
        parser.add_argument("--run_name", default="default_run")
        # import sys
        # sys.argv=[""]
        # args = parser.parse_args()
        self.args = parser.parse_args(sys.argv[1:])
        print(sys.argv)

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

    def log_images_to_wandb(self, train_X, train_Y, model, img_size):
        # Generate reconstructions
        if self.args.use_wandb:

            n_images_to_log = 10  # Right now: cannot be more than #test_imgs = 1000

            model.eval()
            with torch.no_grad():

                X = train_X[:n_images_to_log, :, :]
                Y = train_Y[:n_images_to_log, :, :, :]

                X = torch.from_numpy(X) / 255
                Y = torch.from_numpy(Y) / 255

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

    def train(self):
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
        train_dataloader = load_data(batch_size=self.args.batch_size)
        test_dataloader = load_data(train=False, batch_size=self.args.batch_size)

        # Init model
        if self.args.use_CNN:
            model = ConvVAE()
        else:
            encoder = Encoder(
                input_dim=self.args.fc_flattened_dim,
                fc_hidden_dim=self.args.fc_hidden_dim,
                fc_latent_dim=self.args.fc_latent_dim,
            )
            decoder = Decoder(
                fc_latent_dim=self.args.fc_latent_dim,
                fc_hidden_dim=self.args.fc_hidden_dim,
                output_dim=self.args.fc_flattened_dim,
            )
            model = Net(Encoder=encoder, Decoder=decoder).to(self.DEVICE)

        if self.args.use_wandb:
            wandb.watch(model, log_freq=100)

        optimizer = Adam(model.parameters(), lr=self.args.lr)

        print("Start training VAE...")
        for epoch in range(self.args.n_epochs):
            train_loss = 0
            test_loss = 0
            train_rec = 0
            train_kld = 0

            # ______________TRAIN______________
            model.train()
            for train_i, (X, Y) in tqdm.tqdm(enumerate(train_dataloader)):
                if train_i == 0:
                    X_test = X
                    Y_test = Y

                Y = Y.permute(0, 3, 1, 2)
                X = resize(X, (img_size, img_size))
                Y = resize(Y, (img_size, img_size))

                if self.args.use_CNN:
                    X = X[:, None, :, :]
                else:
                    X = X.view(self.args.batch_size, img_size * img_size)
                    Y = Y.view(self.args.batch_size, img_size * img_size, 2)

                X = X.to(self.DEVICE)
                Y = Y.to(self.DEVICE)

                optimizer.zero_grad()

                X_hat, mean, log_var = model(X)
                rec, kld = self.loss_function2(Y, X_hat, mean, log_var)
                loss = rec + kld

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            # ______________VAL______________
            with torch.no_grad():
                model.eval()
                for eval_i, (X, Y) in tqdm.tqdm(enumerate(test_dataloader)):

                    Y = Y.permute(0, 3, 1, 2)
                    X = resize(X, (img_size, img_size))
                    Y = resize(Y, (img_size, img_size))

                    if self.args.use_CNN:
                        X = X[:, None, :, :]
                    else:
                        X = X.view(self.args.batch_size, self.args.fc_flattened_dim)
                        Y = Y.view(self.args.batch_size, self.args.fc_flattened_dim, 2)

                    X = X.to(self.DEVICE)

                    X_hat, mean, log_var = model(X)
                    rec, kld = self.loss_function2(Y, X_hat, mean, log_var)
                    loss = rec + kld

                    train_rec += rec.item()
                    train_kld += kld.item()
                    test_loss += loss.item()

            # Wandb
            if self.args.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss / (train_i * self.args.batch_size),
                        "test_loss": test_loss / (eval_i * self.args.batch_size),
                        "train_rec": train_rec / (train_i * self.args.batch_size),
                        "train_kld": train_kld / (train_i * self.args.batch_size),
                    }
                )

            # Save current model
            if epoch % 5 == 0:
                save_path = f"models/{self.args.run_name}_model{epoch}.pth"
                torch.save(model.state_dict(), save_path)

            print(
                "\tEpoch",
                epoch + 1,
                "complete!",
                "\tAverage train Loss: ",
                train_loss / (train_i * self.args.batch_size),
                "\tAverage test loss: ",
                test_loss / (eval_i * self.args.batch_size),
            )

        print("Finish training")

        self.log_images_to_wandb(X_test, Y_test, model, img_size)


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


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
