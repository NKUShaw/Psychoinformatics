from collections import OrderedDict
import pandas as pd
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from models import networks
from utils.dataset import FacesMemorabilityDataset, load_data
from utils.args import get_args
from utils.loss import *
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

def predict_and_collect_results(model, device, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().flatten())
    return predictions

if __name__ == '__main__':
    args = get_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    random_seed = 42
    dir_checkpoint = Path(f'./saved_weights/{args.model}')
    labels_file = '../data/Wilma Bainbridge_10k US Adult Faces Database/Full Attribute Scores/final.xlsx' 
    root_dir = '../data/Wilma Bainbridge_10k US Adult Faces Database/10k US Adult Faces Database/Face Images' 

    test_transform = transforms.Compose([transforms.Resize((224, 224)), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
    
    if args.model == 'CNN':
        model = networks.CNN(input_channels=3)
    elif args.model == 'alex':
        model = networks.AlexNet()
    elif args.model == 'MLP':
        model = networks.MLP()
    elif args.model == 'VGG16':
        model = networks.VGG16()
    elif args.model == 'VIT':
        model = networks.VIT()
    else:
        print("还没更新！")
    model = model.to(device)
    dataframe = pd.read_excel(labels_file)
    dataset, _ = train_test_split(dataframe, test_size=0.001, random_state=42)
    dataset = dataset.reset_index(drop=True)
    # _, _, test_df = load_data(labels_file, random_state=42)
    results_df = pd.DataFrame()
    
    for folder in os.listdir(dir_checkpoint):
        folder_path = os.path.join(dir_checkpoint, folder)
        model_target = str(folder)
        print(f'Predicting the {model_target}!')


        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.pth'):
                    file_path = os.path.join(folder_path, file)
                    checkpoint = torch.load(file_path)
                    model.load_state_dict(checkpoint)
                    model.eval()
                    test_dataset = FacesMemorabilityDataset(dataframe=dataset, root_dir=root_dir, target=model_target, transform=test_transform)
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                    predictions = predict_and_collect_results(model, device, test_loader)
                    results_df[model_target] = np.round(predictions, 2)
    results_df.to_excel('./prediction_results.xlsx', index=False)
                    
                    
                    