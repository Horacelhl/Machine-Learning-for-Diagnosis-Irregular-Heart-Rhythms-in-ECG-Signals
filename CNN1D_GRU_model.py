#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, load_model

# define cnn model
def CNN1D_GRU():
    
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=6, activation='relu', input_shape=(3000, 8)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(3), strides=(2), padding="same"))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv1D(filters=64, kernel_size=6, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.2))
              
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))       
    #GRU layer
    model.add(layers.GRU(32, return_sequences=True))
    model.add(layers.Flatten())
    
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(9, activation='softmax'))

    print(model.summary())
    return model

