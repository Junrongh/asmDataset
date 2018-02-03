# INFO: This script will read the images into dictionaries with their features included
# TODO: COMP series, sd series, Ti-0.2pd file are missing

import numpy as np
import scipy.io as sio
import pandas as pd
import sqlite3
import csv

import os
import glob

import skimage
from skimage.color import gray2rgb
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
# import matplotlib.pyplot as plt

import scale

micrographs_info = pd.read_csv('Micrograph_Data.csv',
                               encoding='latin_1', na_values='NaN')

features = ['Image Code', 'Record Name', 'Short Name', 'Material name',
            'UNS number', 'Composition', 'Condition', 'Condition details',
            'Properties', 'Description', 'Photo method', 'Photo details',
            'Etchant', 'Original magnification', 'Image number', 'Keywords',
            'Submit date']

root_path = '../Micrographs'
scaled_root_path = '../Micrographs_scaled'


def read(main, m_missed):
    m_missed = [main + i for i in m_missed]
    main_dic = []
    for i in micrographs_info.index:
        dic = {}
        if micrographs_info.loc[i]['main'] == main:
            if micrographs_info.loc[i]['Image Code'] not in m_missed:
                path = micrographs_info.loc[i]['Image Code'] + '_h.png'
                print('Loading image {}'.format(path))
                for f in features:
                    dic[f] = micrographs_info.loc[i][f]
                dic['main'] = main
                main_dic.append(dic)
    return main_dic


def dic2dtf(overview_dic):

    features_img = ['main', 'Image Code', 'Record Name', 'Short Name',
                    'Material name', 'UNS number', 'Composition', 'Condition',
                    'Condition details', 'Properties', 'Description',
                    'Photo method', 'Photo details', 'Etchant',
                    'Original magnification', 'Image number', 'Keywords',
                    'Submit date']

    info_sorted = pd.DataFrame(columns=features_img)
    for dic in overview_dic:
        n = len(dic)
        df = pd.DataFrame(index=range(0, n), columns=features_img)
        for i in range(0, n):
            for j in features_img:
                df.loc[i][j] = dic[i][j]
        info_sorted = pd.concat([info_sorted, df], ignore_index=True)

    return info_sorted


###############################################################################
# Al(0001~0445)
# 431 info & 433 micrographs
# info missed:          0001, 0297, 0298, 0299, 0300, 0302, 0310,
#                       0311, 0324, 0325, 0326, 0327, 0339, 0340
# micrographs missed:   0434, 0435, 0436, 0437, 0438, 0439, 0440,
#                       0442, 0443, 0443, 0444, 0445
al_info_missed = ['0001', '0297', '0298', '0299', '0300', '0302', '0310',
                  '0311', '0324', '0325', '0326', '0327', '0339', '0340']
al_micrographs_missed = ['0434', '0435', '0436', '0437', '0438', '0439',
                         '0440', '0441', '0442', '0443', '0443', '0444',
                         '0445']

###############################################################################
# As(0001~0163)
# 153 info & 158 micrographs
# info missed:          0026, 0123, 0124, 0139, 0140, 0143, 0150,
#                       0152, 0153, 0161
# micrographs missed:   0123, 0124, 0139, 0140, 0161
as_info_missed = ['0026', '0123', '0124', '0139', '0140', '0143', '0150',
                  '0152', '0153', '0161']
as_micrographs_missed = ['0123', '0124', '0139', '0140', '0161']

###############################################################################
# Cc(0001~0192)
# 191 info & 191 micrographs
# info missed:          0139
# micrographs missed:   0139
cc_info_missed = ['0139']
cc_micrographs_missed = ['0139']

###############################################################################
# Ci(0001~0319)
# 310 info & 309 micrographs
# info missed:          0167, 0224, 0274, 0277, 0278, 0279, 0287
# micrographs missed:   0193, 0311, 0312, 0313, 0314, 0315, 0316,
#                       0317, 0318, 0319
ci_info_missed = ['0167', '0224', '0274', '0277', '0278', '0279', '0287']
ci_micrographs_missed = ['0193', '0311', '0312', '0313', '0314', '0315',
                         '0316', '0317', '0318', '0319']

###############################################################################
# Co(0001~0013)
# 13 info & 4 micrographs
# info missed:
# micrographs missed:   0005, 0006, 0007, 0008, 0009, 0010, 0011,
#                       0012, 0013
co_info_missed = []
co_micrographs_missed = ['0005', '0006', '0007', '0008', '0009', '0010',
                         '0011', '0012', '0013']


###############################################################################
# Cs(0001~1491)
# 1481 info & 1488 micrographs
# info missed:          0256, 0266, 0560, 1222, 1223, 1224, 1420,
#                       1421, 1422, 1423
# micrographs missed:   0265, 0266, 0560, 1482, 1483, 1484, 1485,
#                       1486, 1487, 1488, 1489, 1490, 1491
cs_info_missed = ['0256', '0266', '0560', '1222', '1223', '1224', '1420',
                  '1421', '1422', '1423']
cs_micrographs_missed = ['0265', '0266', '0560', '1482', '1483', '1484',
                         '1485', '1486', '1487', '1488', '1489', '1490',
                         '1491']


###############################################################################
# Cu(0001~0519)
# 459 info & 499 micrographs
# info missed:          0005, 0006, 0007, 0008, 0010, 0012, 0013,
#                       0014, 0015, 0018, 0020, 0055, 0079, 0080,
#                       0108, 0147, 0155, 0156, 0157, 0178, 0222,
#                       0284, 0285, 0370, 0372, 0373, 0405, 0419,
#                       0422, 0424, 0426, 0442, 0444, 0445, 0446,
#                       0447, 0448, 0452, 0454, 0456, 0458, 0460,
#                       0461, 0462, 0463, 0464, 0465, 0466, 0467,
#                       0470, 0471, 0472, 0473, 0474, 0475, 0476,
#                       0477, 0478, 0479, 0480
# micrographs missed:   0055, 0079, 0080, 0108, 0147, 0178, 0222,
#                       0285, 0508, 0509, 0510, 0511, 0512, 0513,
#                       0514, 0515, 0516, 0517, 0518, 0519
cu_info_missed = ['0005', '0006', '0007', '0008', '0010', '0012', '0013',
                  '0014', '0015', '0018', '0020', '0055', '0079', '0080',
                  '0108', '0147', '0155', '0156', '0157', '0178', '0222',
                  '0284', '0285', '0370', '0372', '0373', '0405', '0419',
                  '0422', '0424', '0426', '0442', '0444', '0445', '0446',
                  '0447', '0448', '0452', '0454', '0456', '0458', '0460',
                  '0461', '0462', '0463', '0464', '0465', '0466', '0467',
                  '0470', '0471', '0472', '0473', '0474', '0475', '0476',
                  '0477', '0478', '0479', '0480']
cu_micrographs_missed = ['0055', '0079', '0080', '0108', '0147', '0178',
                         '0222', '0285', '0508', '0509', '0510', '0511',
                         '0512', '0513', '0514', '0515', '0516', '0517',
                         '0518', '0519']

###############################################################################
# Hs(0001~0034)
# 33 info & 21 micrographs
# info missed:          0012
# micrographs missed:   0012, 0023, 0024, 0025, 0026, 0027, 0028,
#                       0029, 0030, 0031, 0032, 0033, 0034
hs_info_missed = ['0012']
hs_micrographs_missed = ['0012', '0023', '0024', '0025', '0026', '0027',
                         '0028', '0029', '0030', '0031', '0032', '0033',
                         '0034']

###############################################################################
# Lz(0001~0066)
# 62 info & 66 micrographs
# info missed:          0027, 0029, 0031, 0046
# micrographs missed:
lz_info_missed = ['0027', '0029', '0031', '0046']
lz_micrographs_missed = []

###############################################################################
# Mg(0001~0066)
# 65 info & 53 micrographs
# info missed:          0042
# micrographs missed:   0054, 0055, 0056, 0057, 0058, 0059, 0060,
#                       0061, 0062, 0063, 0064, 0065, 0066
mg_info_missed = ['0042']
mg_micrographs_missed = ['0054', '0055', '0056', '0057', '0058', '0059',
                         '0060', '0061', '0062', '0063', '0064', '0065',
                         '0066']

###############################################################################
# Ni(0001~0049)
# 48 info & 48 micrographs
# info missed:          0023
# micrographs missed:   0023
ni_info_missed = ['0023']
ni_micrographs_missed = ['0023']

###############################################################################
# Pl(0001)
# 1 info & 1 micrographs
# info missed:
# micrographs missed:
pl_info_missed = []
pl_micrographs_missed = []

###############################################################################
# Rf(0001~0055)
# 55 info & 55 micrographs
# info missed:
# micrographs missed:
rf_info_missed = []
rf_micrographs_missed = []

###############################################################################
# Sc(0001~0059)
# 51 info & 59 micrographs
# info missed:          0025, 0027, 0031, 0033, 0036, 0053, 0056,
#                       0057
# micrographs missed:
sc_info_missed = ['0025', '0027', '0031', '0033', '0036', '0053', '0056',
                  '0057']
sc_micrographs_missed = []

###############################################################################
# Sp(0001~0176)
# 170 info & 149 micrographs
# info missed:          0003, 0099, 0101, 0102, 0107, 0108
# micrographs missed:   0099, 0107, 0108, 0153, 0154, 0155, 0156,
#                       0157, 0158, 0159, 0160, 0161, 0162, 0163,
#                       0164, 0165, 0166, 0167, 0168, 0169, 0170,
#                       0171, 0172, 0173, 0174, 0175, 0176
sp_info_missed = ['0003', '0099', '0101', '0102', '0107', '0108']
sp_micrographs_missed = ['0099', '0107', '0108', '0153', '0154', '0155',
                         '0156', '0157', '0158', '0159', '0160', '0161',
                         '0162', '0163', '0164', '0165', '0166', '0167',
                         '0168', '0169', '0170', '0171', '0172', '0173',
                         '0174', '0175', '0176']

###############################################################################
# Ss(0001~0131)
# 126 info & 127 micrographs
# info missed:          0088, 0089, 0090, 0091, 0098
# micrographs missed:   0088, 0089, 0090, 0091
ss_info_missed = ['0088', '0089', '0090', '0091', '0098']
ss_micrographs_missed = ['0088', '0089', '0090', '0091']

###############################################################################
# Ti(0001~0314)
# 301 info & 276 micrographs
# info missed:          0018, 0081, 0082, 0103, 0104, 0194, 0204
# micrographs missed:   0172, 0277, 0278, 0279, 0280, 0281, 0282,
#                       0283, 0284, 0285, 0286, 0287, 0288, 0289,
#                       0290, 0291, 0292, 0293, 0294, 0295, 0296,
#                       0297, 0298, 0299, 0300, 0301, 0302, 0303,
#                       0304, 0305, 0306, 0307, 0308, 0309, 0310,
#                       0311, 0312, 0313, 0314
ti_info_missed = ['0018', '0081', '0082', '0103', '0104', '0194', '0204']
ti_micrographs_missed = ['0172', '0277', '0278', '0279', '0280', '0281',
                         '0282', '0283', '0284', '0285', '0286', '0287',
                         '0288', '0289', '0290', '0291', '0292', '0293',
                         '0294', '0295', '0296', '0297', '0298', '0299',
                         '0300', '0301', '0302', '0303', '0304', '0305',
                         '0306', '0307', '0308', '0309', '0310', '0311',
                         '0312', '0313', '0314']

###############################################################################
# Ts(0001~0026)
# 24 info & 24 micrographs
# info missed:          0016, 0017
# micrographs missed:   0016, 0017
ts_info_missed = ['0016', '0017']
ts_micrographs_missed = ['0016', '0017']

###############################################################################
# Un(0001~0028, 0501~0509)
# 36 info & 35 micrographs
# info missed:          0010
# micrographs missed:   0010, 0509
un_info_missed = ['0010']
un_micrographs_missed = ['0010', '0509']


al_dic = read('al', al_micrographs_missed)
as_dic = read('as', as_micrographs_missed)
cc_dic = read('cc', cc_micrographs_missed)
ci_dic = read('ci', ci_micrographs_missed)
co_dic = read('co', co_micrographs_missed)
cs_dic = read('cs', cs_micrographs_missed)
cu_dic = read('cu', cu_micrographs_missed)
hs_dic = read('hs', hs_micrographs_missed)
lz_dic = read('lz', lz_micrographs_missed)
mg_dic = read('mg', mg_micrographs_missed)
ni_dic = read('ni', ni_micrographs_missed)
pl_dic = read('pl', pl_micrographs_missed)
rf_dic = read('rf', rf_micrographs_missed)
sc_dic = read('sc', sc_micrographs_missed)
sp_dic = read('sp', sp_micrographs_missed)
ss_dic = read('ss', ss_micrographs_missed)
ti_dic = read('ti', ti_micrographs_missed)
ts_dic = read('ts', ts_micrographs_missed)
un_dic = read('un', un_micrographs_missed)

overview_dic = [cs_dic]

result = dic2dtf(overview_dic)

