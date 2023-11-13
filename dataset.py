import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import os
import pickle

# local
from utils import sentence_embedding



class MyDataset(Dataset):
    def __init__(self, split):
        assert split in ['train','test']
        self.split = split
        self.df = pd.read_csv(f'E:/TravelCompetition/{split}.csv')
        self.targets = pd.read_csv('E:/TravelCompetition/train.csv')['cat3'].unique()
        self.category_dict = {category: idx for idx, category in enumerate(self.targets)}
        self.resize = A.Compose([A.Resize(256,256)])
        if not os.path.exists(f'./txt_embedding/{split}_txt_embedding.pkl'):
            sentence_embedding(split)
        with open(f'./txt_embedding/{split}_txt_embedding.pkl', 'rb') as f:
            self.txt_embedding = pickle.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        # image
        img = cv2.imread(f'E:/TravelCompetition/{self.df.iloc[idx,1][2:]}', cv2.IMREAD_COLOR)
        img = self.resize(image=img)['image']
        img = np.einsum('...c->c...', img)
        img = torch.from_numpy(img).type(torch.FloatTensor)

        # text
        txt = self.txt_embedding[idx]
        txt = torch.from_numpy(txt).type(torch.FloatTensor)
        
        # label
        if self.split == 'train':
            label = self.df.iloc[idx,5]
            index_label = torch.LongTensor([self.category_dict[label]])
            return txt, img, index_label
        else:
            return txt, img
    




if __name__ == '__main__':
    split = 'test'
    dataset = MyDataset(split)
    dataloader = DataLoader(dataset, 2, True)
    if split == 'train':
        txt, img, label = next(iter(dataloader))
        print(f'{txt.shape}, {img.shape}, {label.shape}')
    else:
        txt, img = next(iter(dataloader))
        print(f'{txt.shape}, {img.shape}')