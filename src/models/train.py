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

from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.models.model import Encoder, Decoder, Model


class Trainer:

    def __init__(self):
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--n_epochs", default=5, type=int)
        parser.add_argument("--batch_size", default=100, type=int)
        parser.add_argument("--x_dim", default=784, type=float)
        parser.add_argument("--latent_dim", default=20, type=float)
        parser.add_argument("--hidden_dim", default=400, type=float)

        parser.add_argument("--use_wandb", default=True, type=bool)
        parser.add_argument("--plot_results", default=True, type=bool)
        parser.add_argument("--use_cuda", default=False, type=bool)

        parser.add_argument("--dataset_path", default="/Users/heshe/Desktop/mlops/cookiecutter_project/data")
        parser.add_argument("--run_name", default="default_run")

        self.args = parser.parse_args(sys.argv[2:])
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

        # Init data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=False)

        # Init model
        encoder = Encoder(input_dim=self.args.x_dim, hidden_dim=self.args.hidden_dim, latent_dim=self.args.latent_dim)
        decoder = Decoder(latent_dim=self.args.latent_dim, hidden_dim=self.args.hidden_dim, output_dim=self.args.x_dim)
        model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

        optimizer = Adam(model.parameters(), lr=self.args.lr)

        print("Start training VAE...")
        model.train()
        for epoch in range(self.args.n_epochs):
            overall_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(self.args.batch_size, self.args.x_dim)
                x = x.to(DEVICE)

                optimizer.zero_grad()

                x_hat, mean, log_var = model(x)
                loss = self.loss_function(x, x_hat, mean, log_var)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()
            print("\tEpoch", epoch + 1, "complete!",
                  "\tAverage Loss: ", overall_loss / (batch_idx * self.args.batch_size))

        print("Finish!!")

        # Generate reconstructions
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(test_loader):
                x = x.view(self.args.batch_size, self.args.x_dim)
                x = x.to(DEVICE)
                x_hat, _, _ = model(x)
                break

        res_imgs_path = "/Users/heshe/Desktop/mlops/image-restoration/reports/figures/"
        save_image(x.view(self.args.batch_size, 1, 28, 28), res_imgs_path + "orig_data.png")
        save_image(x_hat.view(self.args.batch_size, 1, 28, 28), res_imgs_path + "reconstructions.png")

        # Generate samples
        with torch.no_grad():
            noise = torch.randn(self.args.batch_size, self.args.latent_dim).to(DEVICE)
            generated_images = decoder(noise)

        save_image(generated_images.view(self.args.batch_size, 1, 28, 28), res_imgs_path + "generated_sample.png")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
