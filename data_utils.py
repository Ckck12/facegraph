import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LandmarkVideoDataset(Dataset):
    def __init__(self, data, dataset_type='train'):
        self.df = pd.read_csv(data).query('train_type == @dataset_type')
                
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        id = str(data['data_name']) +'_'+ str(data['video_id'])
        
        patch_video = np.load(data['patches_npy_path'])
        landmark_idx = np.load(data['landmark_npy_path'])
        landmark_idx = landmark_idx.squeeze(1)
        x3d_vector = np.load(data['x3d_npy_path'])
        # x3d_vector = x3d_vector.squeeze(0)
        label = data['label']

        # pv, l, gv, id, y
        return patch_video, landmark_idx, x3d_vector, id, label
    
# csv_file = '/media/NAS/USERS/moonbo/faceGraph/data/small.csv'
# train_type = 'train'
# dataset = VideoLandmarkDataset(csv_file, train_type)

# print("first data:", dataset[0][0].shape)
# print("first data:", dataset[0][1].shape)
# print("first data:", dataset[0][2].shape)
# print("first data:", dataset[0][3])
# print("first data:", dataset[0][4])
