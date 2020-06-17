import torch
from datetime import datetime

# dataset path
DATA_DF_PATH = './dataset/'
DATA_IMG_PATH = '../dataset/kaggle_rsna(only600)/'

# hyper parameter
MODEL_NAME = 'SequentialGRU'
LOSS_NAME = 'BCE' # BCEWithLogitsLoss
EPOCHS = 100
BATCH_SIZE = 4
INITIAL_LR = 0.001

# etc path
MODEL_WEIGHTS_SAVE_PATH = './checkpoints/rnn'
MODEL_WEIGHTS_LOAD_PATH = './checkpoints/rnn'
TENSORBOARD_PATH = './tensorboard/rnn'

# gpu settings
IS_GPU_PARALLEL = True if torch.cuda.device_count()>1 else False

# train id
timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
TRAIN_ID = f'{timestamp}_{MODEL_NAME}_LR{INITIAL_LR}_BS{BATCH_SIZE}_{LOSS_NAME}Loss'