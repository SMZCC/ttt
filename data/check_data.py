# coding=utf-8
# date: 2018-8-22,13:16:16
# name: smz

import pickle


def check_data_pkl(data_path):
    """
    vot-otb.pkl中仅仅是记录了每个年份的vot下有哪些序列
    :param data_path:
    :return:
    """
    with open(data_path, 'r') as f:
        data = pickle.load(f)
        print 'checking...'


if __name__ == '__main__':
    data_path = './vot-otb-smz.pkl'
    check_data_pkl(data_path)