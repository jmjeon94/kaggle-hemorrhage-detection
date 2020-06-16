from utils.checkpoints import load_checkpoint
from constants import *
from models.densenet121 import DenseNet121_change_avg
from dataloader.dataloader import HmDataset
from dataloader.transforms import build_transform
from main.fit import fit

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

# get test data loader
transforms = build_transform()

train_dataset = HmDataset(df_path='./dataset/train.csv', transforms=transforms, mode=DATASET_MODE)
valid_dataset = HmDataset(df_path='./dataset/valid.csv', transforms=transforms, mode=DATASET_MODE)
test_dataset = HmDataset(df_path='./dataset/test.csv', transforms=transforms, mode=DATASET_MODE)

train_loader = DataLoader(train_dataset,
                         batch_size=1,
                         shuffle=True,
                         num_workers=4)
valid_loader = DataLoader(valid_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)
test_loader = DataLoader(test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1)

# create model
model = DenseNet121_change_avg()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9)

model =  DenseNet121_change_avg()
model, _, _, epoch = load_checkpoint('./checkpoints/cnn/200615_135016_DenseNet121_LR0.001_BS64_BCELoss/030.pth', model, optimizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# test 평가
with torch.no_grad():
    
    # test_loss = fit('Test', epoch, model, test_loader, optimizer, criterion, device)

    for i, (filename, targets, inputs) in enumerate(tqdm(test_loader, position=0, leave=True)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        preds, feas = model(inputs)
