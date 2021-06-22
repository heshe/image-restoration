import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
# from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
# sys.path.insert(0,"C:/Users/Asger/OneDrive/Skrivebord/DTU/Machine_Learning_Operations/image-restoration")


from azureml.core import Workspace, Dataset
# Azure specific
subscription_id = 'e532aafd-37bb-4999-a15d-df27df5b4fae'
resource_group = 'mlops'
workspace_name = 'mlops'

workspace = Workspace(subscription_id, resource_group, workspace_name)

def load_dataset(path="", train=True, small_dataset=False):
    print("Loading data... \n")
    dataset = Dataset.get_by_name(workspace, name='image-resto')

    ROOT = str(Path(__file__).parent.parent.parent)
    dataset.download(target_path= ROOT + '/data/raw', overwrite=False)

    path = ROOT + '/data/raw'
    gray_imgs = np.load(path + "/gray_scale.npy")
    gray_tensor = torch.from_numpy(gray_imgs)

    ab1 = np.load(path + "/ab1.npy")
    ab2 = np.load(path + "/ab2.npy")
    ab3 = np.load(path + "/ab3.npy")
    ab12 = np.append(ab1, ab2, axis=0)
    ab = np.append(ab12, ab3, axis=0)
    ab_tensor = torch.from_numpy(ab)

    # ____________ Process ___________
    gray_processed = gray_tensor / 255
    ab_processed = ab_tensor / 255

    if small_dataset:
        ten_pct_mark = int(ab_processed.size()[0] * (0.1))
        gray_processed = gray_processed[:ten_pct_mark, :, :]
        ab_processed = ab_processed[:ten_pct_mark, :, :]

    cut_amount = int(ab_processed.size()[0] * (0.9))

    if train:
        train_X = gray_processed[:cut_amount, :, :]
        train_Y = ab_processed[:cut_amount, :, :, :]
        return train_X, train_Y
    else:
        test_X = gray_processed[cut_amount:, :, :]
        test_Y = ab_processed[cut_amount:, :, :, :]
        return test_X, test_Y


class mlopsDataset(Dataset):
    def __init__(self, train, path, small_dataset):
        self.gray, self.ab = load_dataset(
            path=path, train=train, small_dataset=small_dataset
        )

    def __len__(self):
        return self.gray.size()[0]

    def __getitem__(self, idx):
        X = self.gray[idx]
        y = self.ab[idx]
        return X, y


def load_data(train=True, path=None, small_dataset=False, batch_size=64, shuffle=True):
    data = mlopsDataset(train, path, small_dataset)
    return DataLoader(data, batch_size, shuffle)


if __name__ == "__main__":
    load_dataset()
