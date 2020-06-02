from utils.checkpoints import load_checkpoint
from constants import *
from models.densenet121 import DenseNet121_change_avg
from dataloader.dataloader import HmDataset
from dataloader.transforms import transforms
from main.fit import fit

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn



# get test data loader
test_dataset = HmDataset(df_path='./dataset/test.csv', transforms=transforms, mode=DATASET_MODE)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         drop_last=False,
                         num_workers=4)

# create model
model = DenseNet121_change_avg()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9)

model =  DenseNet121_change_avg()
model, optimizer, _, epoch = load_checkpoint('./checkpoints/200601_223249_DenseNet121_LR0.001_BS64_BCELoss/  8.pth', model, optimizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# test 평가
with torch.no_grad():
    test_loss = fit('Test', epoch, model, test_loader, optimizer, criterion, device)

print(test_loss)



