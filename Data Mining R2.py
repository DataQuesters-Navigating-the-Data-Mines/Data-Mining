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


# # Exploratory Data Analyis
# 

# In[61]:


fig, ax = plt.subplots(figsize=(20, 15))
mask=np.triu(np.ones_like(df.drop(columns=['DATE_DIED']).corr()))
sns.heatmap(df.drop(columns=['DATE_DIED']).corr(), mask = mask, annot = True, cmap = "Blues", vmin = -1, vmax = 1)
plt.title('Data Correlation', color = 'black', fontsize = 30)
plt.show()


# In[65]:


df_med = df.drop(columns=['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'DATE_DIED','INTUBED', 'AGE', 'CLASIFFICATION_FINAL', 'ICU'], axis=1)
df_med


# In[66]:


df_med.duplicated(keep = False).sum()


# In[68]:


dict(df['DEAD'].value_counts())[1]


# In[74]:


D = df['DEAD']
D = D.replace(1, "DEAD")
D = D.replace(2, "Alive")
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%f'))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Deaths Count', color = '#800000', fontsize = 20)
sns.countplot(x=D, palette = ['#990000', '#0a75ad'])
plt.xlabel('Dead or Alive', fontsize=15)
plt.ylabel('Count', fontsize=15)


# In[79]:


labels = ['Alive', 'Dead']
sizes = df['DEAD'].value_counts()
colors = ['#0a75ad', '#990000']
plt.figure(figsize = (10, 10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20, 'color': "black"}, colors = colors, explode=[0.05, 0.05])
plt.title('Death Percentage', color = '#800000', fontsize = 30)
plt.legend(df['DEAD'].value_counts(), loc = 'lower right', title = 'Death Rate')
plt.show()


# In[80]:


df_dead = df[df["DEAD"] == 1]
df_dead


# In[83]:


df_dead["CLASIFFICATION_FINAL"].value_counts()


# In[96]:


def Covid_or_Not(val):
    if val >= 4:
        return "Patient is not a Covid 19 Carrier"
    else:
        return "Patient is a Covid 19 Carrier"


# In[97]:


df_dead['Covid_or_Not'] = df_dead["CLASIFFICATION_FINAL"].apply(Covid_or_Not)


# In[98]:


df_dead


# In[95]:


labels = ['Carriers', 'Non Carriers']
sizes = df_dead['Covid_or_Not'].value_counts()
colors = ['#468499', '#ff7373']
plt.figure(figsize = (10, 10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20}, colors = colors)
plt.title('Covid Carriers Percentage among Dead Patients', color = 'Maroon', fontsize = 25)
plt.legend(df_dead['Covid_or_Not'].value_counts(), loc = 'lower right', title = 'Covid Carriers')
plt.show()


# In[94]:


df_dead["Covid_or_Not"].value_counts()


# In[105]:


df['Covid_or_Not'] = df["CLASIFFICATION_FINAL"].apply(Covid_or_Not)


# In[106]:


labels = ["Non Carriers", "Carriers"]
sizes = df['Covid_or_Not'].value_counts()
colors = ['#ff7373', '#468499']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20}, colors = colors)
plt.title('Covid Carriers Percentage among All Patients', color = 'Maroon', fontsize = 30)
plt.legend(df['Covid_or_Not'].value_counts(), loc = 'lower right', title = 'Covid Carriers')
plt.show()


# In[107]:


df["Covid_or_Not"].value_counts()


# In[113]:


Covid_deaths = df[(df['Covid_or_Not'] == "A Covid 19 Carrier")]


# In[114]:


Covid_deaths["DEAD"].value_counts()


# In[121]:


labels = ["Alive", "Dead"]
sizes = Covid_deaths['DEAD'].value_counts()
colors = ['#0a75ad', '#990000']
plt.figure(figsize = (10, 10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20, 'color':"black"}, colors = colors)
plt.title('Death Percentage among Covid Carriers',color = 'red',fontsize = 30)
plt.legend(Covid_deaths['DEAD'].value_counts(), loc = 'lower right', title = 'Death Rate')
plt.show()


# In[122]:


sns.histplot(x=df.AGE, kde = True, color = 'blue')
plt.title('Age Distribution', color = 'blue', fontsize = 20)
plt.show()


# In[123]:


plt.figure(figsize=(15,8))
sns.lineplot(data=df, x="AGE", y="Covid_or_Not")
plt.title('Effect of Age on Covid Classification', color = 'blue', fontsize = 30)
plt.show()


# In[124]:


df_mod = df.copy()


# In[125]:


df_mod['OBESITY'] = ['Obese' if i==1 else "Not Obese" for i in df.OBESITY]


# In[127]:


plt.figure(figsize=(15,8))
sns.countplot(data=df_mod, x="SEX", hue="Covid_or_Not", palette = ['#468499', '#ff7373'])
plt.title('Effect of Gender on Covid Classification', color = 'black', fontsize = 30)
plt.show()


# In[128]:


df_preg = df[df['SEX'] == 1]
df_preg['PREGNANT'].value_counts()


# In[130]:


df_preg['PREGNANT'] = ['Pregnant' if i==1 else 'Non Pregnant' for i in df_preg.PREGNANT]


# In[131]:


plt.figure(figsize=(15,8))
sns.countplot(data=df_preg, x="PREGNANT", hue="Covid_or_Not", palette = ['#468499', '#ff7373'])
plt.title('Effect of Pregnancy on Covid Classification', color = 'black', fontsize = 30)
plt.show()


# In[132]:


df_preg[(df_preg['PREGNANT'] == 'Pregnant')]['Covid_or_Not'].value_counts()


# In[136]:


labels = ["Not a Covid 19 Carrier", "A Covid 19 Carrier"]
sizes = df_preg[(df_preg['PREGNANT'] == 'Pregnant')]['Covid_or_Not'].value_counts()
colors = ['#ff7373', '#468499']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20},
        colors = colors,)
plt.legend(df_preg[(df_preg['PREGNANT'] == 'Pregnant')]['Covid_or_Not'].value_counts(), loc = 'lower left',
           title = 'Covid Carriers')
plt.title('Covid Carriers among Pregnant Females', color = 'maroon', fontsize = 25)
plt.show()


# In[138]:


df_preg[(df_preg['PREGNANT'] == "Non Pregnant")]['Covid_or_Not'].value_counts()


# In[139]:


labels = ["Not a Covid 19 Carrier", "A Covid 19 Carrier"]
sizes = df_preg[(df_preg['PREGNANT'] == 'Non Pregnant')]['Covid_or_Not'].value_counts()
colors = ['#ff7373', '#468499']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20},
        colors = colors,)
plt.legend(df_preg[(df_preg['PREGNANT'] == "Non Pregnant")]['Covid_or_Not'].value_counts(), loc = 'lower left',
           title = 'Covid Carriers')
plt.title('Covid Carriers among Non Pregnant Females', color = 'maroon', fontsize = 25)
plt.show()


# In[140]:


df_med


# In[141]:


df_diseases = df_med.drop(columns = ["PREGNANT", "OBESITY", 'DEAD'])


# In[148]:


plt.figure(figsize=(20, 25))
index = 1
for i in df_diseases.columns:
    plt.subplot(5, 2, index)
    df_diseases[i] = ["Yes" if j==1 else "No" for j in df_diseases[i]]
    sns.countplot(data=df_diseases, x=i, hue=df["Covid_or_Not"], palette = ['#468499', '#ff7373'])
    index += 1
plt.show()


# In[150]:


plt.figure(figsize=(20, 25))
index = 1
for i in df_diseases.columns:
    plt.subplot(5, 2, index)
    sns.countplot(data=df_diseases, x=i, hue=df['CLASIFFICATION_FINAL'],)

    index += 1
plt.show()


# In[151]:


df.columns


# In[152]:


df['PATIENT_TYPE'].value_counts()


# In[153]:


labels = ["Not Hospitalized", "Hospitalized"]
sizes = df['PATIENT_TYPE'].value_counts()
colors = ['#e13433', '#46549d']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20},
        colors = colors,)
plt.title('Hospitalized Patients',color = 'black',fontsize = 30)
plt.legend(df['PATIENT_TYPE'].value_counts(), loc = 'lower right', title = 'Hospitalized Patients')
plt.show()


# In[154]:


df_hosp = df[df['PATIENT_TYPE']==2]


# In[155]:


df_hosp['DEAD'] = df_hosp['DEAD'].replace(1, 'Dead')
df_hosp['DEAD'] = df_hosp['DEAD'].replace(0, 'Alive')


# In[156]:


df_hosp['DEAD'].value_counts()


# In[157]:


labels = ["Alive", "Dead"]
sizes = df_hosp['DEAD'].value_counts()
colors = ['#0a75ad', '#990000']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20, 'color':"black"}, colors = colors,
        explode=[0.1, 0.1])
plt.title('Death Rate among Hospitalized People', color = 'red', fontsize = 30)
plt.legend(df_hosp['DEAD'].value_counts(), loc = 'lower right', title = 'Death Rate')
plt.show()


# In[158]:


print(df_diseases.shape)
print(df_diseases.columns)


# In[159]:


print(df_hosp.shape)
print(df_hosp.columns)


# In[161]:


df_diseases2 = df_hosp.drop(columns = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'DATE_DIED', 'INTUBED', 'AGE', 'PREGNANT', 'OBESITY', 'CLASIFFICATION_FINAL', 'ICU', 'DEAD', 'Covid_or_Not'])


# In[162]:


plt.figure(figsize=(20, 25))
index = 1
for i in df_diseases2.columns:
    plt.subplot(5, 2, index)
    df_diseases2[i] = ["Yes" if j==1 else "No" for j in df_diseases2[i]]
    sns.countplot(data=df_diseases2, x=i, hue=df_hosp['DEAD'],
                  palette = [ '#990000', '#0a75ad'])
    index += 1
plt.show()


# In[163]:


df_dead.PATIENT_TYPE.value_counts()


# In[164]:


labels = ["Hospitalized", "Not Hospitalized"]
sizes = df_dead.PATIENT_TYPE.value_counts()
colors = ['#46549d', '#e13433']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20}, colors = colors)
plt.title('Hospitalized Patients Percentage among the Dead', color = 'Maroon', fontsize = 30)
plt.legend(df_dead.PATIENT_TYPE.value_counts(), loc = 'lower right', title = 'Hospitalized Patients')
plt.show()


# In[165]:


df[(df['PATIENT_TYPE'] == 2) & (df['ICU'] == 1)]


# In[166]:


df.ICU.value_counts()


# In[167]:


df[(df['PATIENT_TYPE'] == 2)]['ICU'].value_counts()


# In[168]:


labels = ["Not admitted", "Admitted to ICU"]
sizes = df[(df['PATIENT_TYPE'] == 2)]['ICU'].value_counts()
colors = [ '#e13433', '#46549d']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20},
        colors = colors,)
plt.title('ICU Admitted Patients',color = 'black',fontsize = 30)
plt.legend(df[(df['PATIENT_TYPE'] == 2)]['ICU'].value_counts(), loc = 'lower right',
           title = 'ICU Patients')
plt.show()


# In[169]:


df[(df['CLASIFFICATION_FINAL'] < 4) & (df['ICU'] == 1)]


# In[170]:


16858 - 10449


# In[171]:


df[(df['ICU'] == 1)]['CLASIFFICATION_FINAL'].value_counts()


# In[172]:


df[(df['ICU'] == 1)]['Covid_or_Not'].value_counts()


# In[174]:


labels = ["A Covid 19 Carrier", "Not a Covid 19 Carrier"]
sizes = df[(df['PATIENT_TYPE'] == 2)]['Covid_or_Not'].value_counts()
colors = ['#468499', '#ff7373']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20},
        colors = colors,)
plt.legend(df[(df['PATIENT_TYPE'] == 2)]['ICU'].value_counts(), loc = 'lower left',
           title = 'Covid Carriers')
plt.title('Covid Carriers among ICU Patients', color = 'maroon', fontsize = 25)
plt.show()


# In[175]:


df[(df['ICU'] == 1)]['DEAD'].value_counts()


# In[176]:


labels = ["Alive", "Dead"]
sizes = df[(df['ICU'] == 1)]['DEAD'].value_counts()
colors = [ '#0a75ad', '#990000']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize':20, 'color':"black"}, colors = colors,
        explode=[0.1, 0.1])
plt.title('Death Rate among ICU Patients', color = 'red', fontsize = 25)
plt.legend(df[(df['ICU'] == 1)]['DEAD'].value_counts(), loc = 'lower right', title = 'Death Rate')
plt.show()


# In[177]:


df.columns


# In[183]:


plt.figure(figsize=(15, 8))
sns.histplot(data=df, x='DATE_DIED')
plt.title('Death Trend through Time', color = 'red', fontsize = 25)


# In[ ]:




