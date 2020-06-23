from utils.checkpoints import load_checkpoint
from models.SequenceModel import SequenceModel
from dataloader.dataloader import SequentialHmData, make_pad_sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import pandas as pd
from config.default_config import get_cfg_defaults

# get config
cfg = get_cfg_defaults()
cfg.merge_from_file('config/config_rnn.yaml')
cfg.freeze()
print('Checkpoint load path: ', cfg.REPORT.WEIGHTS_LOAD_PATH)

# dataset 생성
test_dataset = SequentialHmData(feature_path=cfg.DATASET.TEST_FEATURE_PATH, df_path=cfg.DATASET.TEST_PATH)

# dataloader 생성
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=make_pad_sequence)

# get device type
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create model
model = SequenceModel()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).to(device))
optimizer = optim.Adam(model.parameters())

model, _, _, epoch = load_checkpoint(cfg.REPORT.WEIGHTS_LOAD_PATH, model, optimizer)
model.to(device)
model.eval()

# test 평가
wLoss = []

with torch.no_grad():

    submission = []
    type_ = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

    for i, (p_labels, p_features, targets, filenames) in enumerate(tqdm(test_loader, position=0, leave=True)):

        # get data
        p_labels, p_features, targets = p_labels.to(device), p_features.to(device), targets.to(device)

        # inference
        _, pred_seq2 = model(p_features.float(), p_labels.float())

        # cuda to cpu(numpy)
        pred_seq2 = torch.sigmoid(pred_seq2).squeeze(3).squeeze(0).transpose(0, 1).cpu().detach().numpy()
        targets = targets.squeeze(3).squeeze(0).transpose(0, 1).cpu().detach().numpy()

        for label, fn in zip(pred_seq2, filenames[0]):

            for i, l in enumerate(label):
                # filename, label
                row = [fn+'_'+type_[i], l]
                submission.append(row)

    # dataframe생성 후 저
    df = pd.DataFrame(submission, columns=['ID', 'Label'])
    df.to_csv('dataset/submission.csv', index=False)


