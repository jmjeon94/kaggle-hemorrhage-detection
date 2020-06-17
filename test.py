from utils.checkpoints import load_checkpoint
from utils.metrics import print_metrics
from constants import *
from models.densenet121 import DenseNet121_change_avg
from dataloader.dataloader import HmDataset
from dataloader.transforms import build_transform

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
import numpy as np

# get transforms
transforms = build_transform()

# get dataset
train_dataset = HmDataset(df_path='./dataset/train.csv', transforms=transforms, mode=DATASET_MODE)
valid_dataset = HmDataset(df_path='./dataset/valid.csv', transforms=transforms, mode=DATASET_MODE)
test_dataset = HmDataset(df_path='./dataset/test.csv', transforms=transforms, mode=DATASET_MODE)

# get dataloader
train_loader = DataLoader(train_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=4)
valid_loader = DataLoader(valid_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4)
test_loader = DataLoader(test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=1)

# create model
model = DenseNet121_change_avg()

# get device type
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set loss function, optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device))
optimizer = optim.Adam(model.parameters())

# load model weights
model, _, _, epoch = load_checkpoint('./checkpoints/cnn/200616_182029_DenseNet121_LR0.0005_BS64_BCELoss/034.pth', model, optimizer)
model.to(device)
model.eval()

# test 평가
wLoss=[]
with torch.no_grad():
    
    # test_loss = fit('Test', epoch, model, test_loader, optimizer, criterion, device)

    for i, (filename, targets, inputs) in enumerate(tqdm(test_loader, position=0, leave=True)):

        # get data
        inputs, targets = inputs.to(device), targets.to(device)

        # inference
        preds, _ = model(inputs)

        # get weighted bce loss
        loss = criterion(preds, targets).item()
        wLoss.append(loss)

        # cuda to cpu(numpy)
        preds = torch.sigmoid(preds).cpu().detach().numpy().round()
        targets = targets.cpu().detach().numpy()

        if i == 0:
            y_true = targets
            y_pred = preds
        else:
            y_true = np.concatenate([y_true, targets], axis=0)
            y_pred = np.concatenate([y_pred, preds], axis=0)

    print('Loss: {:.2f}'.format(np.mean(wLoss)))
    print_metrics(y_true, y_pred)