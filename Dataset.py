import torch
from torch.utils import data
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class Dataset(data.Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, filename, mode, transform=None):
        super(Dataset, self).__init__()
        # 'Initialization'
        self.data_dir = data_dir
        self.filename = filename
        self.mode = mode
        self.length_data = 0
        self.transform = transform
        data = pd.read_csv(os.path.join(self.data_dir, self.filename), encoding="utf-8")
        self.train_dataset, self.test_dataset = train_test_split(data, test_size=0.2, random_state=42)

        if self.mode == "train":
            self.length_data = len(self.train_dataset)
        else:
            self.length_data = len(self.test_dataset)

    def __len__(self):
        # 'Denotes the total number of samples'
        return self.length_data

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        hour, user, shop, previous_shops, rating = dataset.iloc[index, :]
        hour = torch.tensor(hour)
        user = torch.tensor(user)
        shop = torch.tensor(shop)
        previous_shops = torch.tensor(previous_shops)
        rating = torch.tensor(rating)
        sample = hour, user, shop, \
                 previous_shops,\
                 rating
        return sample


if __name__ == '__main__':
    dataset = Dataset('data_script', 'data_for_model.csv', 'train')
    for sample in dataset:
        print(sample)
        break