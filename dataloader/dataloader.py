from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


# dataset for cnn
class HmDataset(Dataset):
    def __init__(self, df_path, img_dir='../dataset/kaggle_rsna(only600)/imgs/', transforms=None, mode='3ch'):
        self.df = pd.read_csv(df_path)
        self.transforms = transforms
        self.mode = mode
        self.img_dir = img_dir

    def __getitem__(self, index):
        hm_meta = self.df.iloc[index]
        filename = hm_meta.filename
        label = torch.from_numpy(hm_meta['epidural':'any'].values.astype(np.float))

        img = Image.open(self.img_dir + filename + '.png')

        if self.mode == 'sequential':
            study_instance_uid = hm_meta.study_instance_uid
            slice_id_current = hm_meta.slice_id

            # 해당 환자의 DataFrame
            tmp_df = self.df[self.df.study_instance_uid == study_instance_uid]

            # 해당 환자의 slice 전체 수
            max_slice = tmp_df.shape[0]

            # get prev, next slice number
            slice_id_prev = slice_id_current if slice_id_current == 0 else slice_id_current - 1
            slice_id_next = slice_id_current if slice_id_current == max_slice - 1 else slice_id_current + 1

            # get prev, next filename
            filename_prev = tmp_df[tmp_df.slice_id == slice_id_prev].iloc[0].filename
            filename_next = tmp_df[tmp_df.slice_id == slice_id_next].iloc[0].filename

            # get prev, next img
            img_prev = Image.open('../dataset/kaggle_rsna(only600)/imgs/' + filename_prev + '.png')
            img_next = Image.open('../dataset/kaggle_rsna(only600)/imgs/' + filename_next + '.png')

            # concat 3 imgs
            img = np.concatenate([np.expand_dims(np.array(img)[:, :, 0], axis=2),
                                  np.expand_dims(np.array(img_prev)[:, :, 0], axis=2),
                                  np.expand_dims(np.array(img_next)[:, :, 0], axis=2)],
                                 axis=2)

            # From numpy to PIL Image
            img = Image.fromarray(img)

        elif self.mode=='single':
            # get only (40, 80) window image and make it to 3 channels
            img = np.expand_dims(np.array(img)[:,:,0], axis=2)
            img = np.concatenate([img, img, img], axis=2)

            # From numpy to PIL Image
            img = Image.fromarray(img)

        elif self.mode=='3ch':
            # nothing to change
            pass

        else:
            raise ValueError('Not Supported Mode[{}]'.format(self.mode))

        # transform
        if self.transforms is not None:
            img = self.transforms(img)

        return filename, label, img

    def __len__(self):
        return len(self.df)


# dataset for rnn
class SequentialHmData(Dataset):
    def __init__(self, feature_path, df_path):
        self.feature_df = pd.read_csv(feature_path)
        self.ref_df = pd.read_csv(df_path)
        self.person_ids = self.ref_df.study_instance_uid.unique()
        
    def __getitem__(self, index):
        # get patient id
        current_person_id = self.person_ids[index]
        
        # get filenames corresponding with patient id
        filenames = self.ref_df[self.ref_df.study_instance_uid==current_person_id].filename
        df_current_person = self.feature_df[self.feature_df.filename.isin(filenames)]
        
        # get predicted label and features from cnn outputs
        pred_label = df_current_person.iloc[:,1:7].values
        pred_features = df_current_person.iloc[:,7:].values
        
        # get label
        gt_label = self.ref_df[self.ref_df.study_instance_uid==current_person_id].loc[:,'epidural':'any'].values
        
        
        return torch.from_numpy(pred_label), torch.from_numpy(pred_features), torch.from_numpy(gt_label)
    
    def __len__(self):
        return len(self.ref_df.study_instance_uid.unique())
    
def make_pad_sequence(datas):
    
    pred_labels = [data[0] for data in datas]
    pred_features = [data[1] for data in datas]
    gt_labels = [data[2] for data in datas]

    # pad all sequence corresponding with Sequence Length
    pred_labels = pad_sequence(pred_labels, batch_first=True)
    pred_features = pad_sequence(pred_features, batch_first=True)
    gt_labels = pad_sequence(gt_labels, batch_first=True)

    # shape: (N, SL, F) -> (N, SL, 1, F) -> (N, F, SL, 1) 
    pred_labels = pred_labels.unsqueeze(dim=2).permute(0,3,1,2)
    pred_features = pred_features.unsqueeze(dim=2).permute(0,3,1,2)
    gt_labels = gt_labels.unsqueeze(dim=2).permute(0,3,1,2)
        
    return pred_labels, pred_features, gt_labels

