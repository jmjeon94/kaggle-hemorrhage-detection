import torch.nn as nn

def build_loss(loss_type='BCELoss'):

    if loss_type=='BCELoss':
        return nn.BCELoss()

    elif loss_type=='BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()