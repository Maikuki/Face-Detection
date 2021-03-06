# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd
import time
import scipy.io as sio
import os
from tqdm import tqdm

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

face_features = sio.loadmat('cosface.mat')
print('Loaded mat')

sample_sub = open('/media/Data/yangty/arcface/data/submission_template.csv', 'r') 
sub = open('cosface_submission.csv', 'w') 
print('Loaded CSV')

lines = sample_sub.readlines()
pbar = tqdm(total=len(lines))
for line in lines:
    pair = line.split(',')[0]
    sub.write(pair+',')
    a,b = pair.split(':')
    score = '%.2f'%cosin_metric(face_features[a][0], face_features[b][0])
    sub.write(score+'\n')
    pbar.update(1)

sample_sub.close()
sub.close()
