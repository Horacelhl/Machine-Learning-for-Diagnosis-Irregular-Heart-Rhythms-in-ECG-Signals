#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm
#import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


# In[2]:


def get_file(record_base_path):
    files = []
    for filename in os.listdir(record_base_path):
        if filename.endswith('.mat'):
            path = os.path.join(record_base_path, filename)
            files.append(path)
            files = sorted(files)
    return files


# In[3]:


def get_data(path):
    X_list = []
    for i in range(len(path)):    
        ecg = sio.loadmat(path[i])['ECG'][0][0][2][:, : ].T    
        X_list.append(ecg)
    return X_list


# In[4]:


record_base_path = './CPSC2018'
filemat = get_file(record_base_path)
X = get_data(filemat)


# In[5]:

#sftp://hlai@ozstar.swin.edu.au/fred/oz138/COS80028/P1/ConvLSTM1D/CPSC2018/A0001.mat

count=0
for i in tqdm(range(len(X))):    
    fig = plt.figure(frameon=False)
    plt.plot(X[i])       
    filename = 'train_new/' + filemat[i].split('/')[-1].split('.')[0] + ".jpeg";count+=1    
    fig.savefig(filename)
    plt.close(fig)


# from glob import glob
# images = glob("train_new2/*.jpeg")

# import tensorflow as tf
# X = []
# vid_data = []
# for i in tqdm(range(len(images))):
#     image = tf.io.read_file(images[i])
#     image = tf.image.decode_jpeg(image, channels=1)
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, size=[224, 224])
#     datu=np.asarray(image)
#     normu_dat=datu/255
#     vid_data.append(normu_dat)
#     X=np.array(vid_data)

# In[ ]:





# In[ ]:





# In[ ]:




