import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from constants import *
from models.densenet121 import DenseNet121_change_avg
from dataloader.transforms import build_transform
from dataloader.dataloader import HmDataset
from utils.checkpoints import save_checkpoint
from utils.mails import send_mail
from main.fit import fit
from torchsummary import summary

# 모델 생성
model = DenseNet121_change_avg()
# summary(model, (3,512,512), device='cpu')

# transform 생성
transforms = build_transform()

# dataset 생성
train_dataset = HmDataset(df_path='./dataset/train.csv', transforms=transforms, mode=DATASET_MODE)
valid_dataset = HmDataset(df_path='./dataset/valid.csv', transforms=transforms, mode=DATASET_MODE)
test_dataset = HmDataset(df_path='./dataset/test.csv', transforms=transforms, mode=DATASET_MODE)

# dataloader 생성
train_loader = DataLoader(train_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=4)
valid_loader = DataLoader(valid_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4)

# get device type
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# get loss, optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).to(device))
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)

# tensorboard log
writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_PATH, TRAIN_ID))
train_losses = []
valid_losses = []

for epoch in range(1, EPOCHS+1):

    # train
    train_loss = fit('Train', epoch, model, train_loader, optimizer, criterion, device)
    with torch.no_grad():
        valid_loss = fit('Valid', epoch, model, valid_loader, optimizer, criterion, device)
        
    # log
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    # tensor board
    writer.add_scalar('Loss/Train/', train_loss, epoch)
    writer.add_scalar('Loss/Valid/', valid_loss, epoch)
    # writer.add_scalar('Accuracy/Train/', accuracy, epoch+1)

    # save model
    save_checkpoint(epoch, model, optimizer, 'cnn', TRAIN_ID)

    if epoch%10==0:
        send_mail(f'[Epoch:{epoch}]학습 진행중', '')
    
send_mail(f'[알림]학습완료','EC2 종료할 것!!')

