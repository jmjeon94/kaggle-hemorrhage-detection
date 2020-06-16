from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import torch.nn as nn
import numpy as np

def print_metrics(targets, preds, verbose=0):
    '''
    :param targets: gt targets with (N, 6) (dtype:torch.Variable)
    :param preds: prediction with (N, 6) (dtype:torch.Variable)
    :return: print recall, precision, f1 score, accuracy
    '''

    cols =['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    print(f'type\tRecall\tPrec\tF1\tAccuracy')
    for i in range(6):
        r = recall_score(targets[:,i], preds[:,i], zero_division=0)
        p = precision_score(targets[:,i], preds[:,i], zero_division=0)
        acc = accuracy_score(targets[:,i], preds[:,i])
        f1 = f1_score(targets[:,i], preds[:,i], zero_division=0)

        print(f'{cols[i][:5]}:\t{r:.3f}\t{p:.3f}\t{f1:.3f}\t{acc:.3f}')
