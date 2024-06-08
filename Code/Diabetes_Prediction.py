#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('data.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.isnull().sum()


# In[5]:


data.corr()


# In[6]:


import seaborn as sns


# In[7]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot = True)


# In[8]:


X = data.iloc[:,:-1]
y = data['Outcome']


# In[9]:


X.shape


# In[10]:


y.shape


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# In[13]:


print("Train Set: ", X_train.shape, y_train.shape)
print("Test Set: ", X_test.shape, y_test.shape)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)


# In[15]:


from sklearn.metrics import accuracy_score


# In[16]:


print(accuracy_score(y_test, model.predict(X_test))*100)


# In[17]:


import pickle


# In[18]:


pickle.dump(model, open("diabetes.pkl",'wb'))

