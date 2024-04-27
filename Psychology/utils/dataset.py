import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

def load_data(labels_file, random_state=42):
    dataframe = pd.read_excel(labels_file)
    train_val_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=random_state)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df

class FacesMemorabilityDataset(Dataset):
    def __init__(self, dataframe, root_dir, target, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.target = target

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        target_score = self.dataframe.loc[index, self.target]
        if self.transform:
            image = self.transform(image)

        return image, target_score
