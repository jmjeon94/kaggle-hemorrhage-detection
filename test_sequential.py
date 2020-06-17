from utils.checkpoints import load_checkpoint
from utils.metrics import print_metrics
from constants import *
from models.SequenceModel import SequenceModel
from dataloader.dataloader import SequentialHmData, make_pad_sequence

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
import numpy as np

# dataset 생성
train_dataset = SequentialHmData(feature_path='./dataset/train_features.csv', df_path='./dataset/train.csv')
valid_dataset = SequentialHmData(feature_path='./dataset/valid_features.csv', df_path='./dataset/valid.csv')
test_dataset = SequentialHmData(feature_path='./dataset/test_features.csv', df_path='./dataset/test.csv')

# dataloader 생성
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=make_pad_sequence)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=make_pad_sequence)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=make_pad_sequence)

# get device type
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create model
model = SequenceModel()
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).to(device))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

model, _, _, epoch = load_checkpoint('./checkpoints/rnn/200617_165658_SequentialGRU_LR0.001_BS4_BCEWithLogitsLossLoss/030.pth', model, optimizer)
model.to(device)

# test 평가
wLoss = []
with torch.no_grad():
    for i, (p_labels, p_features, targets) in enumerate(tqdm(test_loader, position=0, leave=True)):

        # get data
        p_labels, p_features, targets = p_labels.to(device), p_features.to(device), targets.to(device)

        # inference
        _, pred_seq2 = model(p_features.float(), p_labels.float())

        # reshape to calculate weighted bce loss -> 마지막 axis를 기준으로 weight을 줌
        p = pred_seq2.permute(0, 2, 3, 1)
        t = targets.permute(0, 2, 3, 1).float()

        # get weighted bce loss
        wLoss.append(criterion(p, t).item())

        # cuda to cpu(numpy)
        pred_seq2 = torch.sigmoid(pred_seq2).squeeze(3).squeeze(0).transpose(0, 1).cpu().detach().numpy().round()
        targets = targets.squeeze(3).squeeze(0).transpose(0, 1).cpu().detach().numpy()

        if i == 0:
            y_true = targets
            y_pred = pred_seq2
        else:
            y_true = np.concatenate([y_true, targets], axis=0)
            y_pred = np.concatenate([y_pred, pred_seq2], axis=0)

    print_metrics(y_true, y_pred)
    print('Loss: {:.3f}'.format(np.mean(wLoss)))

