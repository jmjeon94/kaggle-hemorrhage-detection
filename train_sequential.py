import numpy as np
import os

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.mails import send_mail
from utils.checkpoints import save_checkpoint
from utils.metrics import print_metrics

from constants_rnn import *
from models.SequenceModel import SequenceModel
from dataloader.dataloader import SequentialHmData, make_pad_sequence
from main.fit_rnn import fit

# dataset 생성
train_dataset = SequentialHmData(feature_path='./dataset/train_features.csv', df_path='./dataset/train.csv')
valid_dataset = SequentialHmData(feature_path='./dataset/valid_features.csv', df_path='./dataset/valid.csv')
test_dataset = SequentialHmData(feature_path='./dataset/test_features.csv', df_path='./dataset/test.csv')

# dataloader 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=make_pad_sequence)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=make_pad_sequence)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=make_pad_sequence)

# 모델 생성
model = SequenceModel(ch_in=1024)

# get device type
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# get loss, optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9)

# tensorboard log
writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_PATH, TRAIN_ID))
train_losses = []
valid_losses = []

# train
for epoch in range(1, EPOCHS+1):
        
    # fit
    train_loss = fit('Train', epoch, model, train_loader, optimizer, criterion, device)
    with torch.no_grad():
        valid_loss = fit('Valid', epoch, model, valid_loader, optimizer, criterion, device)
        
    # log
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    # tensor board
    writer.add_scalar('Loss/Train/', train_loss, epoch)
    writer.add_scalar('Loss/Valid/', valid_loss, epoch)

    # save model
    save_checkpoint(epoch, model, optimizer, 'rnn', TRAIN_ID)
    
#     if epoch%10==0:
#         send_mail(f'[Epoch:{epoch}]학습 진행중', '')
    
# send_mail(f'[알림]학습완료','EC2 종료할 것!!')



### test
for i, (p_labels, p_features, targets) in enumerate(test_loader):
    
    # get data
    p_labels, p_features, targets = p_labels.to(device), p_features.to(device), targets.to(device)
    
    # inference
    _, pred_seq2 = model(p_features.float(), p_labels.float())
    
    # cuda to cpu(numpy)
    pred_seq2 = pred_seq2.squeeze(3).squeeze(0).transpose(0,1).cpu().detach().numpy().round()
    targets = targets.squeeze(3).squeeze(0).transpose(0,1).cpu().detach().numpy()
    
    if i==0:
        y_true = targets
        y_pred = pred_seq2
    else:
        y_true = np.concatenate([y_true, targets], axis=0)
        y_pred = np.concatenate([y_pred, pred_seq2], axis=0)

print_metrics(y_true, y_pred)
