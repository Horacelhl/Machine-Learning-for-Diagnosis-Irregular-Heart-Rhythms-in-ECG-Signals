#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib as mpl
from Preprocessing import get_data, get_file
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
    plt.savefig(sec_path + 'training_history%d.png'%EPOCHS)
    plt.show()    


# In[3]:


# define cnn model
def Bidirectional_GRU():    
    model = models.Sequential(name="Bidirectional_GRU")
    model.add(layers.Bidirectional(layers.GRU(64, return_sequences=True), input_shape=(input_size)))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.GRU(32)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model


# In[4]:


# define cnn model
def Bidirectional_LSTM():    
    model = models.Sequential(name="Bidirectional_LSTM")
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(input_size)))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dropout(0.5))
    #model.add(layers.Flatten())

    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model


# In[ ]:


# define cnn model
def CNN1D_GRU():    
    model = models.Sequential(name="CNN1D_GRU")
    model.add(layers.Conv1D(filters=64, kernel_size=6, activation='relu', input_shape=(input_size)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(3), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv1D(filters=64, kernel_size=6, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
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
    
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))       
    #GRU layer
    #model.add(layers.GRU(64, return_sequences=True))
    #model.add(layers.Dropout(0.5))
    model.add(layers.GRU(32, return_sequences=True))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))

    print(model.summary())
    return model


# In[ ]:


# define cnn model
def CNN1D_LSTM():
    
    model = models.Sequential(name="CNN1D_LSTM")
    model.add(layers.Conv1D(filters=64, kernel_size=6, activation='relu', input_shape=(input_size)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(3), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv1D(filters=64, kernel_size=6, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
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
    
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5)) 
    
    #LSTM model
    model.add(layers.LSTM(32))
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dropout(0.5))
    #model.add(layers.LSTM(32))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())

    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))

    print(model.summary())
    return model


# In[ ]:


# define cnn model
def CNN1D():
    
    model = models.Sequential(name="CNN1D")
    model.add(layers.Conv1D(filters=64, kernel_size=6, activation='relu', input_shape=(input_size)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(3), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv1D(filters=64, kernel_size=6, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
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
    
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=(2), strides=(2), padding="same"))
    model.add(layers.Dropout(0.5))     
    model.add(layers.Flatten())

    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))

    print(model.summary())
    return model


# In[ ]:


# run the test harness for evaluating a model
def Bi_GRU():

    # define model
    model = Bidirectional_GRU()    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model_checkpoint_callback = ModelCheckpoint(sec_path + 'weight%d.hdf5'%EPOCHS, save_best_only=True, monitor='val_loss', mode='min')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # fit model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback, lr_callback])
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


# run the test harness for evaluating a model
def Bi_LSTM():

    # define model
    model = Bidirectional_LSTM()    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model_checkpoint_callback = ModelCheckpoint(sec_path + 'weight%d.hdf5'%EPOCHS, save_best_only=True, monitor='val_loss', mode='min')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # fit model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback, lr_callback])
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


# run the test harness for evaluating a model
def CNN1DLSTM():

    # define model
    model = CNN1D_LSTM()    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model_checkpoint_callback = ModelCheckpoint(sec_path + 'weight%d.hdf5'%EPOCHS, save_best_only=True, monitor='val_loss', mode='min')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # fit model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback, lr_callback])
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


# run the test harness for evaluating a model
def CNN1DGRU():

    # define model
    model = CNN1D_GRU()    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model_checkpoint_callback = ModelCheckpoint(sec_path + 'weight%d.hdf5'%EPOCHS, save_best_only=True, monitor='val_loss', mode='min')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # fit model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback, lr_callback])
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


# run the test harness for evaluating a model
def CNN1D_model():

    # define model
    model = CNN1D()    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model_checkpoint_callback = ModelCheckpoint(sec_path + 'weight%d.hdf5'%EPOCHS, save_best_only=True, monitor='val_loss', mode='min')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # fit model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback, lr_callback])
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


def lr_schedule(epoch):
    learning_rate = 0.01
    if epoch > 10:
        learning_rate = 0.005
    if epoch > 200:
        learning_rate = 0.001
    if epoch > 300:
        learning_rate = 0.0005
        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


# In[7]:


# load dataset
record_base_path = "./CPSC2018"
filemat = get_file(record_base_path)
X = get_data(filemat)

df = pd.read_csv('./CPSC2018/REFERENCE.csv')
sort = df.sort_values(by=["Recording"], ascending=True)
y = pd.get_dummies(sort["First_label"][:X.shape[0]])

# prepare pixel data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)


EPOCHS = 600
BATCH_SIZE = 32
num_classes = 9
input_size = 3000, 8

print("EPOCHS:", EPOCHS, "input size", input_size)

sec_path = './CNN1D_results/'
CNN1D_model()

sec_path = './CNN1D_LSTM_results/'
CNN1DLSTM()

sec_path = './CNN1D_GRU_results/'
CNN1DGRU()

