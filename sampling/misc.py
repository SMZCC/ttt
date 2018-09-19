# coding=utf-8
# date: 2018-9-14,21:27:19
# name: smz

from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cv2  # 测试函数用


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or 
            2d array of N x [x,y,w,h]
    实现方法：
        1.考虑两个bbox对角线上点4个点
        2.计算重叠部分时: x方向上,左端点取最大,又端点取最小;y方向上, 顶端(对于重叠部分来说的)取最大,底端取最小
        3.计算两个bbox的合并面积: 两个bbox的面积和再减去两个bbox的合并之和
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])     # 左端点取最大
    right = np.minimum(rect1[:, 0]+rect1[:, 2], rect2[:, 0]+rect2[:, 2])  # 右端点取最小
    top = np.maximum(rect1[:, 1], rect2[:, 1])      # 顶端取最大
    bottom = np.minimum(rect1[:, 1]+rect1[:, 3], rect2[:, 1]+rect2[:, 3])  # 底端取最小

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)   # 重叠部分的面积
    union = rect1[:, 2]*rect1[:, 3] + rect2[:, 2]*rect2[:, 3] - intersect     # 两个bbox的合并面积
    iou = np.clip(intersect / union, 0, 1)                                # 计算IoU
    return iou


def crop_image(img, bbox, img_size=107, padding=16, valid=False):
    
    x, y, w, h = np.array(bbox,dtype='float32')

    half_w, half_h = w/2, h/2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w/img_size
        pad_h = padding * h/img_size
        half_w += pad_w
        half_h += pad_h
        
    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)
    
    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)
        
        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8')
        cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
    
    scaled = imresize(cropped, (img_size, img_size))
    return scaled


def change_bbox(bbox, to=None):
    """改变bbox的坐标模式
    args:
        bbox: (x, y, w, h)
        to: 'top-left':以左上角为核心的坐标描述方式
            'center'： 以中心为核心的坐标描述方式
            'four': bbox必须是'top-left'形式的,返回以对角形式描述的bbox
            'four->center: four --> center

              four------------
               ^             |
               |             \/
            top-left <---> center


    return:
        2-dims ndarray  [[], []...],除非没有做任何转变,dim同输入的大小
    """

    if to is None:
        return bbox
    bbox = np.array(bbox)
    if np.ndim(bbox) == 1:
        bbox = bbox[np.newaxis, :]

    assert np.ndim(bbox) < 3, 'Dims of bbox must smaller than 3, SMZ'

    if to == 'center':   # top-left --> center
        changed_bboxes = np.empty((0, 4))
        for bbox_ in bbox:
            cent_x, cent_y = bbox_[:2] + bbox_[2:] / 2
            changed_bboxes = np.concatenate([changed_bboxes,
                                             np.array([[np.round(cent_x), np.round(cent_y),
                                                        bbox_[2], bbox_[3]]])], axis=0)
    elif to == 'top-left':  # center --> top-left
        changed_bboxes = np.empty((0, 4))
        for bbox_ in bbox:
            top_x, top_y = bbox_[:2] - bbox_[2:] / 2
            changed_bboxes = np.concatenate([changed_bboxes,
                                             np.array([[np.round(top_x), np.round(top_y),
                                                        bbox_[2], bbox_[3]]])], axis=0)
    elif to == 'four':   # top-left --> four
        changed_bboxes = np.empty((0, 4))
        for bbox_ in bbox:
            x2, y2 = bbox_[:2] + bbox_[2:]
            changed_bboxes = np.concatenate([changed_bboxes,
                                             np.array([[bbox_[0], bbox_[1],
                                                        np.round(x2), np.round(y2)]])], axis=0)
    elif to == 'four->center':   # four --> center
        changed_bboxes = np.empty((0, 4))
        for bbox_ in bbox:
            width = bbox_[2] - bbox_[0]
            height = bbox_[3] - bbox_[1]
            cent_x = bbox_[0] + width / 2
            cent_y = bbox_[1] + height / 2
            changed_bboxes = np.concatenate([changed_bboxes,
                                             np.array([[cent_x, cent_y, width, height]])], axis=0)
    else:
        return bbox

    return changed_bboxes


def filter_samples(prebbox, samples, ratio=1.6, smz_seq=None):
    """用于过滤掉不符合上一帧结果1.5倍的正样本,可更改
    args:
        prebbox: 上一帧的结果
        samples: 采样的结果
        smz_seq: 用于测试函数用的
    returns:
        prebbox_left_top, holds_4, shows,这三个返回值全部都是left-top类型的bbox
    """
    # change center

    prebbox_center = change_bbox(prebbox, 'center')    # top-left --> center
    new_width, new_height = prebbox[2:] * ratio
    prebbox_center[0][2:] = [np.round(new_width), np.round(new_height)]
    prebbox_top_left = change_bbox(prebbox_center, 'top-left')
    prebbox_four = change_bbox(prebbox_top_left, 'four')    #  top-left ---> four
                                                        # TODO：这里放大bbox的情况可能存在使得bbox超越图像
                                                        # 界限的情况,但是我认为不应该在此处处理这个问题,更应该由
                                                        # 采样部分来解决这个问题,毕竟这里的这个仅仅是为了筛选而用,
                                                        # 而不必进行特征的提取
    samples_four = change_bbox(samples, 'four')   # top-left --- > four

    # compare, holds used to train, shows used to show
    holds = prebbox_four[:, 0] < samples_four[:, 0]   # x1
    holds_1 = samples_four[holds]
    holds = holds == False
    shows_1 = samples_four[holds]

    holds = prebbox_four[:, 1] < holds_1[:, 1]        # y1
    holds_2 = holds_1[holds]
    holds = holds == False
    shows_2 = holds_1[holds]

    holds = prebbox_four[:, 2] > holds_2[:, 2]             # x2
    holds_3 = holds_2[holds]
    holds = holds == False
    shows_3 = holds_2[holds]

    holds = prebbox_four[:, 3] > holds_3[:, 3]             # y2
    holds_4 = holds_3[holds]
    if len(holds_4) == 0:    # 防止一个样本都没有
        holds_3[:, 3] = prebbox_four[:, 3]
        holds_4 = holds_3

    holds = holds == False
    shows_4 = holds_3[holds]


    shows = np.concatenate([shows_1, shows_2, shows_3, shows_4], axis=0)

    # todo: 全部改成top-left
    prebbox_center = change_bbox(prebbox_four, 'four->center')
    prebbox_top_left = change_bbox(prebbox_center, 'top-left')

    holds_4 = change_bbox(holds_4, 'four->center')
    holds_4 = change_bbox(holds_4, 'top-left')

    shows = change_bbox(shows, 'four->center')
    shows = change_bbox(shows, 'top-left')

    return prebbox_top_left, holds_4, shows


def show_samples(image, samples, color='b'):
    """仅仅在一个fig上使用一种显示所有的samples
    show top-left samples
    """
    dpi = 80
    img_shape = image.shape
    fig_size = (img_shape[1]/dpi, img_shape[0]/dpi)
    fig = plt.figure(frameon=False, figsize=fig_size, dpi=dpi)
    ax = plt.Axes(fig, (0., 0., 1., 1.))
    fig.add_axes(ax)
    ax.set_axis_off()
    ax.imshow(image)

    for sample in samples:
        xy = (int(sample[0]), int(sample[1]))
        rect = Rectangle(xy=xy, width=sample[2], height=sample[3],
                         facecolor='none', edgecolor=color, linewidth=1)
        ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    from PIL import Image
    image = Image.open('/media/smz/SMZ_WORKING_SPACE/data_sets/otb_50/Basketball/img/0001.jpg').convert('RGB')
    image = np.asarray(image)
    samples = np.array([[198, 214, 34, 81], [197, 214, 34, 81], [195, 214, 34, 81],
                        [193, 214, 34, 81], [190, 217, 34, 81], [188, 221, 34, 81]])
    # show_samples(image, samples)

    changed_samples = change_bbox(samples, 'center')
    print 'center bboxes:\n', changed_samples
    print 'top-let bboxes:\n', change_bbox(changed_samples, 'top-left')


