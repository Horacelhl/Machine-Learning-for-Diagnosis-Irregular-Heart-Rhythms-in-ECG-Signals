#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import time
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models, Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, UpSampling2D
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
from tensorflow.keras.applications import ResNet50
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[2]:


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    fig = plt.figure(figsize=(16,10))
    
    ## Loss
    fig.add_subplot(2,1,1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    fig.add_subplot(2,1,2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(sec_path + 'training_history_2D_ResNet50_%d.png'%EPOCHS)
    plt.show()    


# In[3]:


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from PIL import Image

images = sorted(glob("train_new/*.jpeg"))
X = []
vid_data = []
for f in tqdm(images):
    image = load_img(f, target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)    
    datu=np.asarray(image)
    normu_dat=datu/255
    vid_data.append(normu_dat)
    X=np.array(vid_data)
    
    


# In[ ]:


df = pd.read_csv('./CPSC2018/REFERENCE.csv')
sort = df.sort_values(by=["Recording"], ascending=True)
y = pd.get_dummies(sort["First_label"][:X.shape[0]])

# prepare pixel data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)


# In[ ]:


resnet_weights_path = './resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

resnet_model = ResNet50(weights=resnet_weights_path, include_top=False, input_shape=(224, 224, 3))
x = UpSampling2D()
x = resnet_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(9, activation='softmax')(x)
model = Model(inputs=resnet_model.input, outputs=output_layer)
for layer in model.layers:
    layer.trainable = False
model.compile(tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# In[ ]:


# compiling the model
EPOCHS = 20
sec_path = './CNN2D_results/'

model_checkpoint_callback = ModelCheckpoint(sec_path + 'weight_2D_ResNet50_%d.hdf5'%EPOCHS, save_best_only=True, monitor='val_loss', mode='min')
es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# fit model
start_time = time.time()
history = model.fit(X_train, y_train, epochs=EPOCHS, 
                    validation_data=(X_test, y_test), batch_size=32, callbacks=[es_callback])#, lr_callback
    
elapsed_time = time.time() - start_time # training time
    
# evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
    
# learning curves    
plot_history(history)
loss = float("{0:.3f}".format(loss))
accuracy = float("{0:.3f}".format(accuracy))
elapsed_time = float("{0:.3f}".format(elapsed_time))

#saving model
hist_df = pd.DataFrame(history.history) 
result = ["Evaluation result: ","loss: "+ str(loss), "accuracy: "+ str(accuracy), "elapsed_time: "+ str(elapsed_time)]
result = pd.DataFrame(result)
with open(sec_path + 'history_2D_ResNet50_%d.csv'%EPOCHS, mode='w') as f:
    result.to_csv(f)
    hist_df.to_csv(f)


# In[ ]:





# In[ ]:




