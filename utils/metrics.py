from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

def print_metrics(targets, preds):
    cols =['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    print(f'type\tRecall\tPrec\tF1\tAccuracy')
    for i in range(6):
        r = recall_score(targets[:,i], preds[:,i])
        p = precision_score(targets[:,i], preds[:,i])
        acc = accuracy_score(targets[:,i], preds[:,i])
        f1 = f1_score(targets[:,i], preds[:,i])

        print(f'{cols[i][:5]}:\t{r:.3f}\t{p:.3f}\t{f1:.3f}\t{acc:.3f}')

