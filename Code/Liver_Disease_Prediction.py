#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('liver.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(0.947064)


# In[8]:


data.isnull().sum()


# Here 2 means suffering with disease and 1 means not suffering with disease.

# Replacing 2 with 1 and 1 with 0, for better understanding.

# In[9]:


data['Dataset'] = data['Dataset'].replace([2,1],[1,0])
data['Dataset'].head()


# In[10]:


data = pd.get_dummies(data, columns = ['Gender'], drop_first = True)


# In[11]:


data.head()


# In[12]:


import seaborn as sns
plt.figure(figsize = (20,20))
sns.heatmap(data.corr(), annot = True)


# In[13]:


sns.pairplot(data, hue = 'Dataset')


# In[14]:


data.corr()


# In[15]:


# X = data[['Albumin_and_Globulin_Ratio', 'Albumin', 'Total_Protiens', 'Aspartate_Aminotransferase', 'Alamine_Aminotransferase', 'Alkaline_Phosphotase', 'Age']]
X = data.drop('Dataset', axis = 1)
y = data['Dataset']


# In[16]:


X.columns


# In[17]:


# from imblearn.combine import SMOTETomek
# smk = SMOTETomek(random_state = 42)
# X, y = smk.fit_sample(X,y)
# X.shape, y.shape


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)


# In[20]:


print("Train Set: ", X_train.shape, y_train.shape)
print("Test Set: ", X_test.shape, y_test.shape)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)


# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[23]:


confusion_matrix(y_test, model.predict(X_test))


# In[24]:


print(f"Accuracy is {round(accuracy_score(y_test, model.predict(X_test))*100,2)}")


# In[25]:


import pickle
pickle.dump(model, open('liver.pkl', 'wb'))

