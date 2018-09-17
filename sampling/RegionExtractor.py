# coding=utf-8
# date: 2018-8-22,13:42:38
# name: smz

import torch
from misc import *
import numpy as np


class RegionExtractor():
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False):

        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(len(samples))   # 给每个sample添加索引
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))  # 每次取出一个batch_size个samples
            index = self.index[self.pointer:next_pointer]   # 选出一个batch_size的样本切片
            self.pointer = next_pointer

            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size,3), dtype='uint8')  # 图片的编码方式为uint8, 0-255
        for i, sample in enumerate(self.samples[index]):   # 注意：index是个切片
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)

        regions = regions.transpose(0,3,1,2).astype('float32')   # ==> (b, c, h, w)
        regions = regions - 128.
        return regions
