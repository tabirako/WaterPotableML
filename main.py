#!/usr/bin/env python
# coding: utf-8

# ## Import needed headers

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# ## Read the files and have a look of what's inside

# In[2]:


df = pd.read_csv('./water_potability.csv', encoding='utf-8', low_memory = True)
df


# ## Making sure that there're no problems with Target Variable

# In[3]:


df.Potability.unique()


# ## Looking at he percetable of yes/no

# In[4]:


df.Potability.value_counts()


# ### The ratio of unpotable water is 1998/total(=3276) = 60.62%

# ## Purge any unfilled cell by setting them to the average

# In[5]:


df['ph'].fillna(float(df['ph'].mean()), inplace=True)
df['Sulfate'].fillna(float(df['Sulfate'].mean()), inplace=True)
df['Trihalomethanes'].fillna(float(df['Trihalomethanes'].mean()), inplace=True)

df.isnull().values.any()


# ## Datas that will be used for the prediction

# In[29]:


X = df.iloc[:, 0:-1]


# In[28]:


y = pd.get_dummies(df["Potability"])


# ## Normalized the X

# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X = scaler.transform(X)


# ## Split the datas into 2 groups, one is the training dataset, another is the testing dataset.

# In[12]:



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,random_state=514,stratify=y)


# In[13]:


X_train


# In[14]:


y_train


# In[15]:


X_test


# In[16]:


y_test


# ## modeling here

# In[18]:



model = tf.keras.Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))


# In[19]:


model.summary()


# In[30]:



model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[22]:


epoch_length = 2000

history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=epoch_length,
                    verbose=0,
                    use_multiprocessing=True,
                    validation_data=(X_test, y_test))


# In[23]:


model.save('water_potability.h5')


# In[34]:


prediction = model.predict(X_test)
prediction = pd.DataFrame(prediction).idxmax(axis=1)
prediction.value_counts()


# ### Verification of the ratio of unpotable water is 198/total(=328) = 60.37%, which is close enough to the 60.62% in population

# In[35]:


history.history.get('loss')
plt.plot(range(epoch_length),history.history.get('loss'))


# In[36]:


plt.plot(range(epoch_length),history.history.get('accuracy'))


# In[40]:


best_score = max(history.history['accuracy'])

print(best_score)


# ## We archieve the best sccre of 98.07% accuracy with 2000 epoches

# In[ ]:




