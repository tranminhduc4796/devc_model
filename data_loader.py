import torch
from torch.utils import data
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class Dataset(data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, filename, mode):
          #'Initialization'
        self.data_dir = data_dir
        self.filename = filename
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.length_data = 0
        
        self.preprocess()
          
        if self.mode == "train":
            self.length_data = len(self.train_dataset)
        else:
            self.length_data = len(self.test_dataset)

    def preprocess(self):
        data = pd.read_csv(os.path.join(self.data_dir,self.filename),encoding = "utf-8")
        self.train_dataset,self.test_dataset = train_test_split(data,test_size = 0.2,random_state=42)
    
    def __len__(self):
        #'Denotes the total number of samples'
        return self.length_data

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        return dataset.loc(index)

def get_loader(data_dir, filename, batch_size = 16, mode = "train", num_workers = 1):
    dataset = Dataset(data_dir, filename, mode)
    data_loader = data.DataLoader(dataset = dataset,
                                  batch_size = batch_size,
                                  shuffle=(mode=="train"),
                                  num_workers = num_workers)
    return data_loader
