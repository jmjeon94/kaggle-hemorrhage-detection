import os
import torch
from constants import *

def save_checkpoint(epoch, model, optimizer, scheduler=None):
    save_folder_dir = os.path.join(MODEL_WEIGHTS_SAVE_PATH, TRAIN_ID)
    if not os.path.exists(save_folder_dir):
        os.makedirs(save_folder_dir, exist_ok=True)
    model_save_path = os.path.join(save_folder_dir, f'{epoch:03d}.pth')

    if IS_GPU_PARALLEL:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    if scheduler is not None:
        scheduler = scheduler.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler
    }, model_save_path)


def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(ckpt_path):
        raise ValueError('No ckpt in [{}]'.format(ckpt_path))

    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model_state_dict'])

    if scheduler is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])

    return model, optimizer, scheduler, epoch
