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

def test(data_loader, model, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    df = pd.DataFrame(data={"Actual": np.array(actuals).flatten(), "Predicted": np.array(predictions).flatten()})
    pearson_corr = df.corr(method='pearson')['Actual']['Predicted']
    print(f"Pearson Correlation of Testset: {pearson_corr}")



if __name__ == '__main__':
    args = get_args()
    # device = 'cuda'
    device = 'cuda:1'
    dir_checkpoint = Path(f'./saved_weights/{args.model}/{args.target}')
    dir_checkpoint.mkdir(parents=True, exist_ok=True)
    best_model_path = dir_checkpoint / 'best_model.pth'
    random_seed = 42

    # 1. Dataloader

    labels_file = '../data/Wilma Bainbridge_10k US Adult Faces Database/Full Attribute Scores/final.xlsx' 
    root_dir = '../data/Wilma Bainbridge_10k US Adult Faces Database/10k US Adult Faces Database/Face Images' 
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
    train_df, val_df, test_df = load_data(labels_file, random_state=random_seed)
    train_dataset = FacesMemorabilityDataset(dataframe=train_df, root_dir=root_dir, target=args.target, transform=train_transform)
    val_dataset = FacesMemorabilityDataset(dataframe=val_df, root_dir=root_dir, target=args.target, transform=train_transform)
    test_dataset = FacesMemorabilityDataset(dataframe=test_df, root_dir=root_dir, target=args.target, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Initialize logging

    logging.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
            Optimizer:       {args.optimizer}
            Momentum:        {args.momentum}
            Weight decay:    {args.weight_decay}
        ''')

    # 3. Obtain model

    if args.model == 'cnn' or args.model == 'CNN':
        model = networks.CNN(input_channels=3)
    elif args.model == 'resnet' or args.model == 'ResNet':
        model = networks.ResNetRegression()
    elif args.model == 'alex':
        model = networks.AlexNet()
    elif args.model == 'MLP':
        model = networks.MLP()
    elif args.model == 'VGG16':
        model = networks.VGG16()
    elif args.model == 'VIT':
        model = networks.VIT()
    elif args.model == 'senet':
        model = se_resnet.se_resnet50(num_classes=1, pretrained=True)
    else:
        print("还没更新！")

    model = model.to(device)


    if args.load is not False:
        pretrain_weight = 'saved_weights' + '/' + args.model + '/' + args.target + '/' + args.load
        print(f"load a pretrained model in {pretrain_weight}.")
        checkpoint = torch.load(pretrain_weight)
        model.load_state_dict(checkpoint)
    else:
        print('未读取任意一种模型')


    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('我还没写')

    if args.loss == 'smoothl1loss':
        criterion = nn.SmoothL1Loss()
    elif args.loss == 'mseloss':
        criterion = nn.MSELoss()
    elif args.loss == 'huberloss':
        criterion = nn.HuberLoss()
    elif args.loss == 'mselosscorr':
        criterion = MSELossCorr(regularization_strength=args.rs)
    else:
        print('我还没写')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    print('==> Building model..')

    # 4. Training
    test(test_loader, model, device)