import torch
from datetime import datetime

# dataset path
DATA_DF_PATH = './dataset/'
DATA_IMG_PATH = '../dataset/kaggle_rsna(only100)/'

# hyper parameter
MODEL_NAME = 'DenseNet121'
LOSS_NAME = 'BCE'
EPOCHS = 100
BATCH_SIZE = 64
INITIAL_LR = 0.001

# etc path
MODEL_WEIGHTS_SAVE_PATH = './checkpoints/'
MODEL_WEIGHTS_LOAD_PATH = './checkpoints/'
TENSORBOARD_PATH = './tensorboard/'

# gpu settings
IS_GPU_PARALLEL = True if torch.cuda.device_count()>1 else False

# train id
timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
TRAIN_ID = f'{timestamp}_{MODEL_NAME}_LR{INITIAL_LR}_BS{BATCH_SIZE}_{LOSS_NAME}Loss'

# dataset mode
DATASET_MODE = 'sequential' # '3ch' or 'sequential'