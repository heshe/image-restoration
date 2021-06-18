import numpy as np
import torch
import sys

#sys.path.insert(0,"C:/Users/Asger/OneDrive/Skrivebord/DTU/Machine_Learning_Operations/image-restoration")

from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Define path to data
path = "C:/Users/Asger/OneDrive/Skrivebord/DTU/Machine_Learning_Operations/data/"

# Store raw data in "data/raw" and processeed in "data/processed"
def load_dataset():
    print("Loading data... \n")
    ab1 = np.load(path + "/ab/ab/ab1.npy")
    ab2 = np.load(path + "/ab/ab/ab2.npy")
    ab3 = np.load(path + "/ab/ab/ab3.npy")
    gray_imgs = np.load(path + "/l/gray_scale.npy")

    ab12 = np.append(ab1, ab2, axis=0)
    ab = np.append(ab12, ab3, axis=0)
    
    ab_tensor = torch.from_numpy(ab)
    gray_tensor = torch.from_numpy(gray_imgs)

    print("Storing raw data... \n")
    torch.save(ab_tensor, "data/raw/ab.pt")
    torch.save(gray_tensor, "data/raw/gray.pt")
    
    #____________ Process ___________
    ab_processed = ab_tensor/255
    gray_processed = gray_tensor/255

    cut_amount = int(ab_processed.size()[0]*(0.9))

    train_X = gray_processed[:cut_amount, :, :]
    test_X = gray_processed[cut_amount:, :, :]

    train_Y = ab_processed[:cut_amount, :, :, :]
    test_Y = ab_processed[cut_amount:, :, :, :]

    print("Storing processed data... \n")
    torch.save(train_X, "data/processed/train_X.pt")
    torch.save(test_X, "data/processed/test_X.pt")
    torch.save(train_Y, "data/processed/train_Y.pt")
    torch.save(test_Y, "data/processed/test_Y.pt")


class mlopsDataset(Dataset):
    def __init__(self, train):
        if train:
            print("Loading train data...")
            self.gray = torch.load("data/processed/train_X.pt")
            self.ab = torch.load("data/processed/train_Y.pt")
        else:
            print("Loading test data...")
            self.gray = torch.load("data/processed/test_X.pt")
            self.ab = torch.load("data/processed/test_Y.pt")

    def __len__(self):
        return self.gray.size()[0]

    def __getitem__(self, idx):
        X = self.gray[idx]
        y = self.ab[idx]
        return X, y

def load_data(train=True, batch_size=64, shuffle=True):
    data = mlopsDataset(train)     
    return DataLoader(data, batch_size, shuffle)


if __name__ == "__main__":
    load_dataset()

        