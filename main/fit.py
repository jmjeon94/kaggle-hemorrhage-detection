from tqdm import tqdm
from constants import *

def fit(phase, epoch, model, data_loader, optimizer, criterion, device):
    losses = 0
    if phase == 'Train':
        model.train()
    elif phase == 'Valid' or phase == 'Test':
        model.eval()

    tbar = tqdm(data_loader, position=0)
    for data in tbar:

        _, target, input_img = data
        target, input_img = target.to(device), input_img.to(device)

        optimizer.zero_grad()

        predicted_label, _ = model(input_img)
        loss = criterion(predicted_label, target.float())

        if phase == 'Train':
            loss.backward()
            optimizer.step()

        #         predicted_label_thresholded = predicted_label>0.5
        #         acc = (predicted_label_thresholded==target).sum() #

        losses += loss.item()

        if phase == 'Train':
            msg = f'[{phase}]\tEpoch:[{epoch}/{EPOCHS}]\tLoss:{losses / len(data_loader):.5f}'
        else:
            msg = f'[{phase}]\tLoss:{losses / len(data_loader):.5f}'
        tbar.set_description(msg)

    return losses / len(data_loader)
