from utils.checkpoints import load_checkpoint
from utils.metrics import print_metrics, weighted_log_loss
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

# create model
model = SequenceModel()
# criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9)

model, _, _, epoch = load_checkpoint('./checkpoints/rnn/200616_143111_SequentialGRU_LR0.001_BS4_BCELoss/030.pth', model,
                                     optimizer)

# get device type
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# test 평가
with torch.no_grad():
    for i, (p_labels, p_features, targets) in enumerate(tqdm(test_loader, position=0, leave=True)):

        # get data
        p_labels, p_features, targets = p_labels.to(device), p_features.to(device), targets.to(device)

        # inference
        _, pred_seq2 = model(p_features.float(), p_labels.float())

        # cuda to cpu(numpy)
        pred_seq2 = pred_seq2.squeeze(3).squeeze(0).transpose(0, 1).cpu().detach().numpy().round()
        targets = targets.squeeze(3).squeeze(0).transpose(0, 1).cpu().detach().numpy()

        if i == 0:
            y_true = targets
            y_pred = pred_seq2
        else:
            y_true = np.concatenate([y_true, targets], axis=0)
            y_pred = np.concatenate([y_pred, pred_seq2], axis=0)

    print_metrics(y_true, y_pred)

