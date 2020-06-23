from yacs.config import CfgNode as CN
import torch
from datetime import datetime

_C = CN()
# ---------------------------------------------------------------------------- #
# CUDA Settings
# ---------------------------------------------------------------------------- #
_C.CUDA = CN()
_C.CUDA.IS_MULTI_GPU = True if torch.cuda.device_count()>1 else False
# ---------------------------------------------------------------------------- #
# Dataset directories
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.ROOT = './dataset/'
_C.DATASET.IMG_PATH = '../dataset/kaggle_rsna(only600)'
# slice dataframe directories
_C.DATASET.TRAIN_PATH = './dataset/train.csv'
_C.DATASET.VALID_PATH = './dataset/valid.csv'
_C.DATASET.TEST_PATH = './dataset/test.csv'
# feature directories
_C.DATASET.TRAIN_FEATURE_PATH = './dataset/train_features.csv'
_C.DATASET.VALID_FEATURE_PATH = './dataset/valid_features.csv'
_C.DATASET.TEST_FEATURE_PATH = './dataset/test_features.csv'
_C.DATASET.MODE_CNN = ''
# ---------------------------------------------------------------------------- #
# Hyper parameters
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.INITIAL_LR = 0.005
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 64
# ---------------------------------------------------------------------------- #
# Model parameters
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.CRITERION = 'BCEWithLogitsLoss'
_C.MODEL.OPTIMIZER = 'Adam'
# ---------------------------------------------------------------------------- #
# Report infos
# ---------------------------------------------------------------------------- #
_C.REPORT = CN()
_C.REPORT.USE_TENSORBOARD = False
_C.REPORT.TENSORBOARD_PATH = './tensorboard/'
_C.REPORT.WEIGHTS_SAVE_PATH = './checkpoints/'
_C.REPORT.WEIGHTS_LOAD_PATH = ''
_C.REPORT.TIMESTAMP = datetime.now().strftime('%y%m%d_%H%M%S')

def get_cfg_defaults():
    return _C.clone()


if __name__=='__main__':

    cfg = get_cfg_defaults()
    cfg.freeze()
    print(cfg)