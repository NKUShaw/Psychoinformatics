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
    spearman_corr = df.corr(method='spearman')['Actual']['Predicted']
    return spearman_corr

if __name__ == '__main__':
    args = get_args()
    device = 'cuda'
    dir_checkpoint = Path(f'./saved_weights/')
    dir_checkpoint.mkdir(parents=True, exist_ok=True)
    best_model_path = f'{dir_checkpoint}/{args.model}_best_model.pth'
    random_seed = 42

    labels_file = '../data/Wilma Bainbridge_10k US Adult Faces Database/Memorability Scores/memorability-scores.xlsx'
    root_dir = '../data/Wilma Bainbridge_10k US Adult Faces Database/10k US Adult Faces Database/Face Images'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_df, test_df = load_data(labels_file, test_size=0.2, random_state=random_seed)
    train_dataset = FacesMemorabilityDataset(dataframe=train_df, root_dir=root_dir, transform=transform)
    test_dataset = FacesMemorabilityDataset(dataframe=test_df, root_dir=root_dir, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
            Optimizer:       {args.optimizer}
            Momentum:        {args.momentum}
            Weight decay:    {args.weight_decay}
        ''')

    if args.model == 'cnn' or args.model == 'CNN':
        model = networks.CNN(input_channels=3)
    elif args.model == 'resnet' or args.model == 'ResNet':
        model = networks.resnet18()
    elif args.model == 'AlexNet':
        model = networks.AlexNet()
    elif args.model == 'MLP':
        model = networks.MLP()
    elif args.model == 'VGG16':
        model = networks.VGG16()
    elif args.model == 'VGG19':
        model = networks.VGG19()
    elif args.model == 'ResNet50':
        model = networks.ResNet50()
    elif args.model == 'ResNet101':
        model = networks.ResNet101()
    elif args.model == 'VIT':
        model = networks.VIT()
    else:
        print("还没更新！")

    model = model.to(device)

    if args.load is not False:
        pretrain_weight = args.load
        checkpoint = torch.load(pretrain_weight)
        model.load_state_dict(checkpoint)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('我还没写')

    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    print('==> Building model..')

    results_df = pd.DataFrame(columns=['Epoch', 'Spearman Correlation'])
    best_spearman_corr = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch} / {args.epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.float()
                targets = targets.float()
                targets = targets.unsqueeze(1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": epoch_loss / (batch_idx + 1)})
                pbar.update(1)
        scheduler.step()
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        current_spearman_corr = test(test_loader, model, device)
        if current_spearman_corr > best_spearman_corr:
            best_spearman_corr = current_spearman_corr
            best_epoch = epoch
            test_corr = round(best_spearman_corr, 2)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch} with Spearman Correlation: {best_spearman_corr}")
            torch.save(model.state_dict(), f'./saved_weights/{args.model}_best_model_{test_corr}.pth')
            # torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            # logging.info(f'Checkpoint {epoch} saved!')
        trainset_corr = test(train_loader, model, device)
        print(
                f"Epoch: {epoch}, \n"
                f"Spearman Correlation of Train Set: {trainset_corr}, \n"
                f"Current Spearman Correlation of Validate Set: {current_spearman_corr}, \n"
                f"Best Spearman Correlation of Validate Set: {best_spearman_corr} (Best Epoch {best_epoch})")