from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

class HmDataset(Dataset):
    def __init__(self, df_path, transforms=None, mode='3ch'):
        self.df = pd.read_csv(df_path)
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, index):
        hm_meta = self.df.iloc[index]
        filename = hm_meta.filename
        label = torch.from_numpy(hm_meta['epidural':'any'].values.astype(np.float))

        img = Image.open('../dataset/kaggle_rsna(only100)/imgs/' + filename + '.png')

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
            img_prev = Image.open('../dataset/kaggle_rsna(only100)/imgs/' + filename_prev + '.png')
            img_next = Image.open('../dataset/kaggle_rsna(only100)/imgs/' + filename_next + '.png')

            # concat 3 imgs
            img = np.concatenate([np.expand_dims(np.array(img)[:, :, 0], axis=2),
                                  np.expand_dims(np.array(img_prev)[:, :, 0], axis=2),
                                  np.expand_dims(np.array(img_next)[:, :, 0], axis=2)],
                                 axis=2)
            img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return filename, label, img

    def __len__(self):
        return len(self.df)
