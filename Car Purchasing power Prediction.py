#!/usr/bin/env python
# coding: utf-8

# # Library import

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


car_data = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')


# In[3]:


car_data.tail(10)


# In[4]:


car_data.head(5)


# In[5]:


sns.pairplot(car_data)


# # Cleaning the data

# In[6]:


X = car_data.drop(['Customer Name','Customer e-mail', 'Country'], axis = 1)


# In[7]:


X


# In[8]:


X= X.drop(['Car Purchase Amount'],axis = 1)


# In[9]:


y= car_data['Car Purchase Amount']
y


# In[10]:


X.shape


# In[11]:


y.shape


# # Data cleaning

# In[12]:


from sklearn.preprocessing import MinMaxScaler


# In[13]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled.shape


# In[14]:


scaler.data_min_


# In[15]:


X_scaled


# In[16]:


y.shape


# In[17]:


y = y.values.reshape(-1,1)


# In[18]:


y_scaled = scaler.fit_transform(y)


# In[19]:


y_scaled


# # Creating test_data and Train_data

# In[20]:


X


# In[21]:


X_scaled.shape


# In[22]:


y_scaled.shape


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size = 0.25)


# In[25]:


X_train.shape


# In[26]:


y_test.shape


# In[27]:


import tensorflow as tf
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(40, input_dim = 5 , activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


# In[28]:


model.summary()


# In[30]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error')
epochs_hist = model.fit(X_train,y_train,epochs = 100,batch_size = 25,verbose =1 ,validation_split = 0.2)


# In[32]:


epochs_hist.history.keys()


# In[38]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title("Loss During Training")
plt.xlabel('Epoch number')
plt.ylabel('Training & Validation Loss')
plt.legend(['Trainging Loss', 'Validation Loss'])


# # Testing The Data

# In[47]:


# Gender, Age, Annual Salary, Credit Card Debt, Net worth
x_test = np.array([[1,50,50000,10000,600000]])
y_predict = model.predict(x_test)


# In[48]:


print("Expected purchase Amount ",y_predict)

