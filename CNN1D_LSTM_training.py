#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from Preprocessing import get_data, get_file
from CNN1D_LSTM import CNN1D_LSTM


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
    plt.savefig(sec_path + 'training_history%d.png'%EPOCHS)
    plt.show()    


# In[3]:


# run the test harness for evaluating a model
def run_test_harness():
	# load dataset 
    filemat = get_file(record_base_path)
    X = get_data(filemat)
    df = pd.read_csv('./CPSC2018/REFERENCE.csv')
    y = pd.get_dummies(df["First_label"][:X.shape[0]])

	# prepare pixel data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
	
    # define model
    model = CNN1D_LSTM()    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model_checkpoint_callback = ModelCheckpoint(sec_path + 'weight%d.hdf5'%EPOCHS, save_best_only=True, monitor='val_loss', mode='min')
    
    # fit model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback])
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
    with open(sec_path + 'history%d.csv'%EPOCHS, mode='w') as f:
        result.to_csv(f)
        hist_df.to_csv(f)


# In[ ]:


# entry point, run the test harness
record_base_path = "./CPSC2018"
sec_path = './LSTM_results/'

EPOCHS = 200
BATCH_SIZE = 64

run_test_harness()


# In[ ]:





# In[ ]:





# In[ ]:




