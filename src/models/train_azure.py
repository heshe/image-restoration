"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import argparse
import numpy as np
import cv2

from azureml.core import Dataset, Run
from PIL import Image
from pathlib import Path
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.optim import Adam

from src.models.model_FC import Encoder, Decoder, Model


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

        parser.add_argument("--run_name", default="default_run")
        
        parser.add_argument('--input-data', type=str, dest='dataset_folder', help='data mount point')

        parser.add_argument("--save_model", default=True, type=bool)
        parser.add_argument("--model_name", default="image_resto", type=str)
        parser.add_argument("--make_reconstructions", default=False, type=bool)

        self.args = parser.parse_args(sys.argv[1:])
        print(sys.argv)

        # Set root path
        self.ROOT = str(Path(__file__).parent.parent.parent)

    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD

    def train(self):
        # Get the experiment run context
        run = Run.get_context() 

        # Training device
        DEVICE = torch.device("cuda" if self.args.use_cuda and torch.cuda.is_available() else "cpu")

        # Get train and test
        # load the diabetes dataset
        print("Loading Data...")
        train_path = run.input_datasets['image_resto']
        ab_imgs = np.load(train_path + "/ab1.npy")
        gray_imgs = np.load(train_path + "/gray_scale.npy")[:10000, :, :]

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
            for i in range(int(train_X.shape[0]/self.args.batch_size)):
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
                
            run.log("Average Loss: ", np.float(overall_loss / (i * self.args.batch_size)))
        
        # Log model and performance
        run.log('LR', np.float(self.args.lr))
        run.log('Epochs', np.int(self.args.n_epochs))
        run.log('Latent dim', np.int(self.args.latent_dim))
        run.log('Hidden dim', np.int(self.args.hidden_dim))
        run.log('Overall loss', np.float(overall_loss))
        print("Finish!!")

        # Generate reconstructions
        if self.args.make_reconstructions:
            model.eval()
            with torch.no_grad():
                #for batch_idx, (x, _) in enumerate(test_loader):
                for i in range(int(test_X.shape[0]/self.args.batch_size)):
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

            res_imgs_path = os.path.join(self.ROOT, "reports", "figures")

            X = X.view(16, 224*224)
            Y = Y.view(16, 224*224, 2)

            X_origin = get_rbg_from_lab((X*255).view(16, 224, 224), (Y*255).view(16, 224, 224, 2), n=10)
            X_hat = get_rbg_from_lab((X*255).view(16, 224, 224), (X_hat*255).view(16, 224, 224, 2), n=10)

            for i in range(X_origin.shape[0]):
                im = Image.fromarray(X_origin[i])
                im.save(os.path.join(res_imgs_path, "orig", f"img{i}.png"))
                im = Image.fromarray(X_hat[i])
                im.save(os.path.join(res_imgs_path, "recon", f"img{i}.png"))

        # Save model
        if self.args.save_model:
            # Save the trained model
            model_file = self.args.model_name + ".pkl"
            joblib.dump(value=model, filename=model_file)
            run.upload_file(
                name = os.path.join(self.ROOT, "models", model_file),
                path_or_stream = './' + model_file,
            )
        
            run.complete()
            # Register the model
            run.register_model(
                model_path=os.path.join(self.ROOT, "models", model_file),
                model_name=self.args.model_name,
                tags={'Training context':'Inline Training'},
                #properties={
                #    'LR': run.get_metrics()['LR'],
                #    'Epochs': run.get_metrics()['Epochs'],
                #    'Latent dim': run.get_metrics()['Latent dim'],
                #    'Hidden dim': run.get_metrics()['Hidden dim'],
                #    'Overall loss': run.get_metrics()['Overall loss'],
                #}
            )
        else:
            run.complete()



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

