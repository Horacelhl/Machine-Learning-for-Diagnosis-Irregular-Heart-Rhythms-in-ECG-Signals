#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm


# In[2]:


def get_file(record_base_path):
    files = []
    for filename in os.listdir(record_base_path):
        if filename.endswith('.mat'):
            path = os.path.join(record_base_path, filename)
            files.append(path)
            files = sorted(files)
    return files


# In[6]:


def get_data(path):
    X_list = []    
    for i in tqdm(range(6877)):
        ecg = np.zeros((72000, 12), dtype=np.float32)
        ecg[-sio.loadmat(path[i])['ECG'][0][0][2].shape[1]:,:] = sio.loadmat(path[i])['ECG'][0][0][2][:, -72000: ].T
        #ecg = sio.loadmat(filemat[i])['ECG'][0][0][2][:, : ].T    
        X_list.append(ecg)        
        #min_batch.append(ecg) 
    return np.array(X_list)


# record_base_path = "./CPSC2018"
# # load dataset 
# filemat = get_file(record_base_path)
# X = get_data(filemat)

# In[ ]:




