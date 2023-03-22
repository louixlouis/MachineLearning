import os
import numpy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets
from torchvision.utils import save_image

if __name__=='__main__':
    onehot = torch.zeros(2, 2)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
    fixed_y_ = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(torch.LongTensor).squeeze()
    print(fixed_y_.shape)
    fixed_y_label_ = onehot[fixed_y_]
    print(fixed_y_label_.shape)

    y_ = (torch.rand(64, 1) * 2).type(torch.LongTensor).squeeze()
    print((torch.rand(4, 1)*2).type(torch.LongTensor))
    print(onehot[y_].shape)
    print(y_)

    with open('gender_label.pkl', 'rb') as fp:
        y_gender_ = pickle.load(fp)

    y_gender_ = torch.LongTensor(y_gender_).squeeze()
    print(y_gender_.shape)