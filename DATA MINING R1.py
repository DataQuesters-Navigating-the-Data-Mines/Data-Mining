#!/usr/bin/env python
# coding: utf-8

# # DATA PREPARATION AND PREPROCESSING

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df = pd.read_csv("Covid Data.csv")


# In[3]:


df.head()


# In[4]:


df


# In[5]:


(df.duplicated().sum()/df.shape[0])*100


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


df_null = df.copy()
for i in [97, 98, 99]:
    df_null.replace(i, np.nan, inplace = True)


# In[9]:


df_null.isnull().sum()


# In[10]:


sns.heatmap(df_null.isnull())
plt.title('Before Handling the Missing Values', color = 'black', fontsize = 15)
plt.show()


# In[11]:


df['DATE_DIED'][df[
    'DATE_DIED'].apply(lambda x: isinstance(x, str))]


# In[12]:


df['DEAD'] = [0 if i=='9999-99-99' else 1 for i in df.DATE_DIED]


# In[13]:


df['DEAD'].value_counts()


# In[14]:


df['DATE_DIED'].replace('9999-99-99', np.nan, inplace = True)


# In[15]:


df['DATE_DIED']


# In[16]:


df['DATE_DIED'] = pd.to_datetime(df['DATE_DIED'])


# In[17]:


df['DATE_DIED']


# In[18]:


df['DATE_DIED'].isnull().sum()


# In[19]:


df.describe().round(3).T.drop('count', axis = 1)


# In[20]:


df['AGE'][df['AGE']> 110].value_counts().sum()


# In[21]:


df.SEX.value_counts()


# In[22]:


df.SEX.shape


# In[23]:


df[(df['SEX'] ==1)].shape


# In[24]:


df[(df['SEX'] == 1)]['PREGNANT']


# In[25]:


df[(df['SEX'] == 1)]['PREGNANT'].value_counts()


# In[26]:


513179+8131+3754


# In[27]:


df[(df['SEX'] == 2)]['PREGNANT']


# In[28]:


df[(df['SEX'] == 2) & (df['PREGNANT'])]['PREGNANT'].value_counts()


# In[29]:


df['PREGNANT'].value_counts()


# In[30]:


df['PREGNANT'].replace (97, 2, inplace = True)


# In[31]:


df['PREGNANT'].value_counts()


# In[32]:


df.ICU.value_counts()


# In[33]:


for i in [1, 2, 97, 99]:
    for j in [1, 2]:
        print(f"At PATIENT_TYPE = {j} and at ICU = {i} the shape will be:", "\n")
        print(df[(df['PATIENT_TYPE'] == j) & (df['ICU'] == i)].shape, "\n", "\n\n")


# In[34]:


df['ICU'].replace (97, 2, inplace = True)


# In[35]:


df.ICU.value_counts()


# In[36]:


df.INTUBED.value_counts()


# In[43]:


for i in [1, 2, 97, 99]:
    for j in [1, 2]:
        print (f"At PATIENT_TYPE = {j} and at INTUBED = {i} the shape will be:", "\n")
        print (df[(df['PATIENT_TYPE'] == j) & (df['INTUBED'] == i)].shape, "\n", "\n\n")


# In[45]:


df['INTUBED'].replace (97, 2, inplace = True)


# In[46]:


df.INTUBED.value_counts()


# In[47]:


for i in [98, 99]:
    df.replace(i, np.nan, inplace = True)


# In[48]:


df


# In[49]:


df_null2 = df.copy()
df_null2.DATE_DIED = df_null2.DATE_DIED.fillna("9999-99-99")


# In[50]:


df_null2.isnull().sum()


# In[52]:


sns.heatmap(df_null2.isnull())
plt.title('After Handling the missing values', color = 'black', fontsize = 15)
plt.show()


# In[54]:


df.describe().round(3).T.drop('count', axis = 1)


# In[56]:


for i in df.columns:
    if(i!='AGE' and i!='DATE_DIED'):
        print(i, "->", dict(df[i].value_counts())


# In[57]:


df

