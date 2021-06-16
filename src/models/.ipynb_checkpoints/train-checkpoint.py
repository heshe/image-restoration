"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import argparse
import numpy as np
import tqdm
import cv2

from PIL import Image
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.optim import Adam

from src.models.model import Encoder, Decoder, Model


class Trainer:

    def __init__(self):
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--n_epochs", default=5, type=int)
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--x_dim", default=224*224, type=float)
        parser.add_argument("--latent_dim", default=20, type=float)
        parser.add_argument("--hidden_dim", default=400, type=float)

        parser.add_argument("--use_wandb", default=True, type=bool)
        parser.add_argument("--plot_results", default=True, type=bool)
        parser.add_argument("--use_cuda", default=False, type=bool)

        parser.add_argument("--dataset_path", default="/Users/heshe/Desktop/mlops/cookiecutter_project/data")
        parser.add_argument("--run_name", default="default_run")

        self.args = parser.parse_args(sys.argv[1:])
        print(sys.argv)

    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD

    def train(self):
        # Training device
        DEVICE = torch.device("cuda" if self.args.use_cuda else "cpu")

        # Data loading
        mnist_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = MNIST(
            self.args.dataset_path + "/MNIST_train", transform=mnist_transform, train=True, download=True
        )
        test_dataset = MNIST(
            self.args.dataset_path + "/MNIST_test", transform=mnist_transform, train=False, download=True
        )

        # Get train and test
        train_path = "/Users/heshe/Desktop/mlops/image-restoration/data/raw/"
        ab_imgs = np.load(train_path + "/ab/ab/ab1.npy")
        gray_imgs = np.load(train_path + "/l/gray_scale.npy")[:10000, :, :]

        train_X = gray_imgs[:9000, :, :]
        test_X = gray_imgs[9000:, :, :]

        train_Y = ab_imgs[:9000, :, :, :]
        test_Y = ab_imgs[9000:, :, :, :]

        # Init data loaders
        #train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)
        #test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=False)

        # Init model
        encoder = Encoder(input_dim=self.args.x_dim, hidden_dim=self.args.hidden_dim, latent_dim=self.args.latent_dim)
        decoder = Decoder(latent_dim=self.args.latent_dim, hidden_dim=self.args.hidden_dim, output_dim=self.args.x_dim)
        model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

        optimizer = Adam(model.parameters(), lr=self.args.lr)

        print("Start training VAE...")
        model.train()
        for epoch in range(self.args.n_epochs):
            overall_loss = 0
            #for batch_idx, (x, _) in enumerate(train_loader):
            for i in tqdm.tqdm(range(int(train_X.shape[0]/self.args.batch_size))):
                l1 = i*self.args.batch_size
                l2 = i*self.args.batch_size + self.args.batch_size
                X = train_X[l1:l2, :, :]
                Y = train_Y[l1:l2, :, :, :]

                X = torch.from_numpy(X)/255
                Y = torch.from_numpy(Y)/255

                X = X.view(self.args.batch_size, self.args.x_dim)
                Y = Y.view(self.args.batch_size, self.args.x_dim, 2)

                X = X.to(DEVICE)

                optimizer.zero_grad()

                X_hat, mean, log_var = model(X)
                loss = self.loss_function(Y, X_hat, mean, log_var)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()
            print("\tEpoch", epoch + 1, "complete!",
                  "\tAverage Loss: ", overall_loss / (i * self.args.batch_size))

        print("Finish!!")

        # Generate reconstructions
        model.eval()
        with torch.no_grad():
            #for batch_idx, (x, _) in enumerate(test_loader):
            for i in tqdm.tqdm(range(int(test_X.shape[0]/self.args.batch_size))):
                l1 = i*self.args.batch_size
                l2 = i*self.args.batch_size + self.args.batch_size
                X = test_X[l1:l2, :, :]
                Y = test_Y[l1:l2, :, :, :]

                X = torch.from_numpy(X)/255
                Y = torch.from_numpy(Y)/255

                X = X.view(self.args.batch_size, self.args.x_dim)
                Y = Y.view(self.args.batch_size, self.args.x_dim, 2)

                X = X.to(DEVICE)
 
                X_hat, _, _ = model(X)
                break

        res_imgs_path = "/Users/heshe/Desktop/mlops/image-restoration/reports/figures/"

        X = X.view(16, 224*224)
        Y = Y.view(16, 224*224, 2)

        X_origin = get_rbg_from_lab((X*255).view(16, 224, 224), (Y*255).view(16, 224, 224, 2), n=10)
        X_hat = get_rbg_from_lab((X*255).view(16, 224, 224), (X_hat*255).view(16, 224, 224, 2), n=10)

        for i in range(X_origin.shape[0]):
            im = Image.fromarray(X_origin[i])
            im.save(f"/Users/heshe/Desktop/mlops/image-restoration/reports/figures/orig/img{i}.png")
            im = Image.fromarray(X_hat[i])
            im.save(f"/Users/heshe/Desktop/mlops/image-restoration/reports/figures/recon/img{i}.png")


def get_rbg_from_lab(gray_imgs, ab_imgs, n = 10):
    
    #create an empty array to store images
    imgs = np.zeros((n, 224, 224, 3))
    
    imgs[:, :, :, 0] = gray_imgs[0:n:]
    imgs[:, :, :, 1:] = ab_imgs[0:n:]
    
    #convert all the images to type unit8
    imgs = imgs.astype("uint8")
    
    #create a new empty array
    imgs_ = []
    
    for i in range(0, 1):
        imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))

    #convert the image matrix into a numpy array
    imgs_ = np.array(imgs_)

    #print(imgs_.shape)
    
    return imgs_


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
