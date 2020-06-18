from tqdm import tqdm
from constants_rnn import *

def fit(phase, epoch, model, data_loader, optimizer, criterion, device):
    losses = 0
    if phase == 'Train':
        model.train()
    elif phase == 'Valid' or phase == 'Test':
        model.eval()

    tbar = tqdm(data_loader, position=0, leave=True)
    for i, (pred_label, pred_feature, gt_label) in enumerate(tbar):

        pred_label, pred_feature, gt_label = pred_label.to(device), pred_feature.to(device), gt_label.to(device)

        optimizer.zero_grad()

        logit1, logit2 = model(pred_feature.float(), pred_label.float())

        # reshape to get weighted criterion -> 마지막 axis를 기준으로 weight를 줌
        logit1 = logit1.permute(0, 2, 3, 1)
        logit2 = logit2.permute(0, 2, 3, 1)
        gt_label = gt_label.permute(0, 2, 3, 1)

        loss1 = criterion(logit1, gt_label.float())
        loss2 = criterion(logit2, gt_label.float())

        loss = loss1 + loss2

        if phase == 'Train':
            loss.backward()
            optimizer.step()

        losses += loss.item()

        tbar.set_description(f'[{phase}]\tEpoch:[{epoch}/{EPOCHS}] Loss:{losses / (i + 1):.5f}')  # '\tAcc:{acc:.2%}')

    return losses / len(data_loader)