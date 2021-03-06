# coding=utf-8
# date: 2018-9-14,21:50:15
# name: smz

import os
import numpy as np
import pickle
from collections import OrderedDict


def preprocess_data():
    seq_home = '/media/smz/SMZ_WORKING_SPACE/data_sets/vot/'
    seqlist_path = './vot-otb.txt'
    output_path = './vot-otb-smz.pkl'

    with open(seqlist_path,'r') as fp:
        seq_list = fp.read().splitlines()

    data = OrderedDict()
    for i, seq in enumerate(seq_list):
        img_list = sorted([p for p in os.listdir(seq_home+seq) if os.path.splitext(p)[1] == '.jpg'])  #仅仅保存的文件名而非路径
        gt = np.loadtxt(seq_home+seq+'/groundtruth.txt',delimiter=',')

        assert len(img_list) == len(gt), "Lengths do not match!!"

        if gt.shape[1] == 8:
            x_min = np.min(gt[:,[0, 2, 4, 6]],axis=1)[:,None]
            y_min = np.min(gt[:,[1, 3, 5, 7]],axis=1)[:,None]
            x_max = np.max(gt[:,[0, 2, 4, 6]],axis=1)[:,None]
            y_max = np.max(gt[:,[1, 3, 5, 7]],axis=1)[:,None]
            gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        data[seq] = {'images':img_list, 'gt':gt}

    with open(output_path, 'wb') as fp:
        pickle.dump(data, fp, -1)


if __name__ == '__main__':
    preprocess_data()