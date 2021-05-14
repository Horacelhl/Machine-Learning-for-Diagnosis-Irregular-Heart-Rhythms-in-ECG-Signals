#!/usr/bin/env python
# coding: utf-8

import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm


def get_file(record_base_path):
    files = []
    for filename in os.listdir(record_base_path):
        if filename.endswith('.mat'):
            path = os.path.join(record_base_path, filename)
            files.append(path)
    return files


def get_data(path):
    min_batch = []    
    for i in tqdm(range(6877)):
        data_12 = sio.loadmat(path[i])['ECG'][0][0][2]
        jieduan = data_12[:,i:i+3000]
        if jieduan.shape[1] <3000:
            jieduan = data_12[:,-3000:]
        min_batch.append(jieduan[[1,2,6,7,8,9,10,11],:].T) 
    return np.array(min_batch).reshape(-1,3000,8)






