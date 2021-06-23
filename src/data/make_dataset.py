import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# import sys
# sys.path.insert(0,"C:/Users/Asger/OneDrive/Skrivebord/DTU/Machine_Learning_Operations/image-restoration")
# Define path to data
# path = "/Users/Morten/Downloads/archive"
path = "C:/Users/Asger/OneDrive/Skrivebord/DTU/Machine_Learning_Operations/data"


# Store raw data in "data/raw" and processeed in "data/processed"
def load_dataset(total_load_amount=True):
    print("Loading data... \n")
    ab1 = np.load(path + "/ab/ab/ab1.npy")
    ab2 = np.load(path + "/ab/ab/ab2.npy")
    ab3 = np.load(path + "/ab/ab/ab3.npy")
    gray_imgs = np.load(path + "/l/gray_scale.npy")

    ab12 = np.append(ab1, ab2, axis=0)
    ab = np.append(ab12, ab3, axis=0)

    ab_tensor = torch.from_numpy(ab)
    gray_tensor = torch.from_numpy(gray_imgs)

    ten_pct_mark = int(ab_tensor.size()[0] * (0.1))
    gray_tensor = gray_tensor[:ten_pct_mark, :, :]
    ab_tensor = ab_tensor[:ten_pct_mark, :, :, :]

    print("Storing raw data... \n")
    # torch.save(ab_tensor, "data/raw/ab.pt")
    # torch.save(gray_tensor, "data/raw/gray.pt")

    # ____________ Process ___________
    ab_processed = ab_tensor / 255
    gray_processed = gray_tensor / 255

    cut_amount = int(ab_processed.size()[0] * (0.9))

    train_X = gray_processed[:cut_amount, :, :]
    test_X = gray_processed[cut_amount:, :, :]

    train_Y = ab_processed[:cut_amount, :, :, :]
    test_Y = ab_processed[cut_amount:, :, :, :]

    if total_load_amount:
        print("Storing full size processed data... \n")
        torch.save(train_X, "data/processed/train_X.pt")
        print("Train X")
        torch.save(test_X, "data/processed/test_X.pt")
        print("Test X")
        torch.save(train_Y, "data/processed/train_Y.pt")
        print("Train Y")
        torch.save(test_Y, "data/processed/test_Y.pt")
        print("Test Y")

    else:
        amount = 100
        print("Storing small size processed data... \n")
        torch.save(train_X[:amount, :, :], "data/processed/train_X_small.pt")
        print("Train X")
        torch.save(test_X[amount:, :, :], "data/processed/test_X_small.pt")
        print("Test X")
        torch.save(train_Y[:amount, :, :, :], "data/processed/train_Y_small.pt")
        print("Train Y")
        torch.save(test_Y[amount:, :, :, :], "data/processed/test_Y_small.pt")
        print("Test Y")


class mlopsDataset(Dataset):
    def __init__(self, train, path, full_size):

        if full_size:
            if train:
                print("Loading train data...")
                self.gray = torch.load(path + "/data/processed/train_X.pt")
                self.ab = torch.load(path + "/data/processed/train_Y.pt")
            else:
                print("Loading test data...")
                self.gray = torch.load(path + "/data/processed/test_X.pt")
                self.ab = torch.load(path + "/data/processed/test_Y.pt")
        else:
            if train:
                print("Loading train data...")
                self.gray = torch.load(path + "/data/processed/train_X_small.pt")
                self.ab = torch.load(path + "/data/processed/train_Y_small.pt")
            else:
                print("Loading test data...")
                self.gray = torch.load(path + "/data/processed/test_X_small.pt")
                self.ab = torch.load(path + "/data/processed/test_Y_small.pt")

    def __len__(self):
        return self.gray.size()[0]

    def __getitem__(self, idx):
        X = self.gray[idx]
        y = self.ab[idx]
        return X, y


def load_data(train=True, batch_size=64, shuffle=True, path="", full_size=True):
    data = mlopsDataset(train, path, full_size)
    return DataLoader(
        data,
        batch_size,
        shuffle,
    )


if __name__ == "__main__":
    load_dataset(total_load_amount=False)
