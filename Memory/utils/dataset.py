import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class FacesMemorabilityDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        memorability_score = self.dataframe.iloc[index, 6] - self.dataframe.iloc[index, 7] #train
        if self.transform:
            image = self.transform(image)
        return image, memorability_score


def load_data(labels_file, test_size=0.2, random_state=42):
    dataframe = pd.read_excel(labels_file)
    train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    return train_df, test_df
