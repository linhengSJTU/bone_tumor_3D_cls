import os
import os.path as osp
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import time
import random
import matplotlib.pyplot as plt
from scipy import ndimage
import yaml

config = yaml.load(open('configs/resnet-attention.yaml'))
window_level = config['window_level']
window_width = config['window_width']
class CustomDataset(data.Dataset):
    def __init__(self, name, rescale_size, fold_file, folds, transform=None):
        self.name = name
        self.rescale_size = rescale_size
        self.transform = transform
        path_df = pd.read_csv(fold_file,encoding='gbk')

        if folds:
            path_df = path_df[path_df.Fold.isin(folds)]
            self.path_df = path_df.reset_index(drop=True)
        else:
            self.path_df = path_df

        # if name == 'train':
        #     self.path_df = self.path_df.sample(8)
        # if name == 'valid':
        #     self.path_df = self.path_df.sample(8)
        # if name == 'infer':
        #     self.path_df = self.path_df.sample(20)

    def __len__(self):
        return len(self.path_df)


    def window(self, image, window_width, window_level):
        min_value = window_level - window_width // 2
        max_value = window_level + window_width // 2
        image[image < min_value] = min_value
        image[image > max_value] = max_value
        return image

    def __getitem__(self, index):
        row = self.path_df.iloc[index]
        image_name = row.Path
        mask_name = row.Path_mask
        # if self.name != 'infer':
        #     flip = row.Flip
        # else:
        #     flip = 0

        _label = row.Label
        # if self.name == 'train':
        #     _label = random.randint(0, 1)
        cls_label = np.array(_label).astype(np.float32)

        image_sitkimg = sitk.ReadImage(image_name)
        # mask_sitkimg = sitk.ReadImage(mask_name)

        image_data = sitk.GetArrayFromImage(image_sitkimg)
        image_data = self.window(image_data,window_width,window_level)
        # mask_data = sitk.GetArrayFromImage(mask_sitkimg)
        # mask_data = (mask_data!=0)*1
        # image_data = mask_data

        origin = image_sitkimg.GetOrigin()
        spacing = image_sitkimg.GetSpacing()
        direction = image_sitkimg.GetDirection()
        shape = image_data.shape

        # scale = [self.rescale_size, self.rescale_size, self.rescale_size]
        # image_data = ndimage.interpolation.zoom(image_data, scale, order=0)

        # out = sitk.GetImageFromArray(image_data)
        # sitk.WriteImage(out,'/data/henglin/bone_tumor/customdata.nii')

        image_data = np.expand_dims(image_data, axis=0)


        if self.transform:
            pass

        cls_labels = torch.from_numpy(cls_label).float()
        imgs = torch.from_numpy(image_data).float()

        return imgs, cls_labels, [image_name, origin, spacing, direction, shape]
