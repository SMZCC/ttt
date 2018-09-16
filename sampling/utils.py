# coding=utf-8
# date: 2018-9-14,21:27:19
# name: smz

from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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


def filter_samples(prebbox, samples):
    """用于过滤掉不符合上一帧结果1.5倍的正样本
    args:
        prebbox: 上一帧的结果
        samples: 采样的结果
    """
    # change center
    center = prebbox[:2] + prebbox[2:] / 2
    



def show_samples(image, samples, color='b'):
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
    show_samples(image, samples)

