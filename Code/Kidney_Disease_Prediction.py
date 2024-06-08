#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('kidney_disease.csv')


# In[3]:


data.head()


# ckd=chronic kidney disease

# In[4]:


data.info()


# In[5]:


data.classification.unique()


# In[6]:


data.classification=data.classification.replace("ckd\t","ckd") 


# In[7]:


data.classification.unique()


# In[8]:


data.drop('id', axis = 1, inplace = True)


# In[9]:


data.head()


# In[10]:


data['classification'] = data['classification'].replace(['ckd','notckd'], [1,0])


# In[11]:


data.head()


# In[12]:


data.isnull().sum()


# In[13]:


df = data.dropna(axis = 0)
print(f"Before dropping all NaN values: {data.shape}")
print(f"After dropping all NaN values: {df.shape}")


# In[14]:


df.head()


# In[15]:


df.index = range(0,len(df),1)
df.head()


# In[16]:


for i in df['wc']:
    print(i)


# In[17]:


df['wc']=df['wc'].replace(["\t6200","\t8400"],[6200,8400])


# In[18]:


for i in df['wc']:
    print(i)


# In[19]:


df.info()


# In[20]:


df['pcv']=df['pcv'].astype(int)
df['wc']=df['wc'].astype(int)
df['rc']=df['rc'].astype(float)
df.info()


# In[21]:


object_dtypes = df.select_dtypes(include = 'object')
object_dtypes.head()


# In[22]:


dictonary = {
        "rbc": {
        "abnormal":1,
        "normal": 0,
    },
        "pc":{
        "abnormal":1,
        "normal": 0,
    },
        "pcc":{
        "present":1,
        "notpresent":0,
    },
        "ba":{
        "notpresent":0,
        "present": 1,
    },
        "htn":{
        "yes":1,
        "no": 0,
    },
        "dm":{
        "yes":1,
        "no":0,
    },
        "cad":{
        "yes":1,
        "no": 0,
    },
        "appet":{
        "good":1,
        "poor": 0,
    },
        "pe":{
        "yes":1,
        "no":0,
    },
        "ane":{
        "yes":1,
        "no":0,
    }
}


# In[23]:


df=df.replace(dictonary)


# In[24]:


df.head()


# In[25]:


import seaborn as sns
plt.figure(figsize = (20,20))
sns.heatmap(df.corr(), annot = True, fmt=".2f",linewidths=0.5)


# In[26]:


df.corr()


# In[27]:


X = df.drop(['classification', 'sg', 'appet', 'rc', 'pcv', 'hemo', 'sod'], axis = 1)
y = df['classification']


# In[28]:


X.columns


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


model = RandomForestClassifier(n_estimators = 20)
model.fit(X_train, y_train)


# In[33]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[34]:


confusion_matrix(y_test, model.predict(X_test))


# In[35]:


print(f"Accuracy is {round(accuracy_score(y_test, model.predict(X_test))*100, 2)}%")


# In[36]:


import pickle
pickle.dump(model, open('kidney.pkl', 'wb'))

