#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[50]:


df=pd.read_csv('ECG200_TRAIN.csv',delimiter="  ",header=None)
df


# In[3]:


test_section=df[0]
test_section.value_counts()


# In[23]:


list_of_dataframes=[]
df=df.sort_values(by=[0])#sorting the values. The dataframe gets ordered with all labels with -1 coming before 1.
df1=df.head(30)#Taking the first 30 rows in another dataframe
df2=df.tail(70)#Taking the last 70 rows in another dataframe
index_for_df1=0
index_for_df2=0
for i in range(0,5):#this loop makes a list of dataframes with an apporpriate ratio
    random1=df1.iloc[index_for_df1:index_for_df1+6]#I take 6 lines from the first dataframe
    random2=df2.iloc[index_for_df2:index_for_df2+14]#I take 14 lines from the second dataframe
    index_for_df1+=3
    index_for_df2+=7
    
    folder=pd.concat([random1,random2])
    
    list_of_dataframes.append(folder)


# In[25]:


fold1=list_of_dataframes[0]
fold2=list_of_dataframes[1]
fold3=list_of_dataframes[2]
fold4=list_of_dataframes[3]
fold5=list_of_dataframes[4]


# In[26]:


fold1


# In[59]:


first_train_dataset=pd.concat([fold1,fold2,fold3,fold4])
first_test_dataset=fold5

second_train_dataset=pd.concat([fold1,fold2,fold3,fold5])
second_test_dataset=fold4


third_train_dataset=pd.concat([fold1,fold2,fold4,fold5])
third_test_dataset=fold3


fourth_train_dataset=pd.concat([fold1,fold3,fold4,fold5])
fourth_test_dataset=fold2


fifth_train_dataset=pd.concat([fold2,fold3,fold4,fold5])
fifth_test_dataset=fold1


# In[60]:


train_list=[first_train_dataset,second_train_dataset,third_train_dataset,fourth_train_dataset,fifth_train_dataset]
test_list=[first_test_dataset,second_test_dataset,third_test_dataset,fourth_test_dataset,fifth_test_dataset]


# In[61]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
Accuracy_list=[]
model = KNeighborsClassifier(n_neighbors=3)
for i in range(0,5):
    Y_train=train_list[i]
    Y_train=Y_train.iloc[:,0]
    Y_train=Y_train.to_numpy()
    Y_train=np.where(Y_train==1.0,1,Y_train)#This operation is needed because model.fit does not except float values
    Y_train=np.where(Y_train==-1.0,-1,Y_train)
    
    
    
    X_train=train_list[i]
    X_train=X_train.loc[:,1:98]
    X_train=X_train.to_numpy()


    model.fit(X_train,Y_train)
    
    Y_test=test_list[i]
    Y_test=Y_test.iloc[:,0]
    Y_test=Y_test.to_numpy()
    Y_test=np.where(Y_test==-1.0,-1,Y_test)
    Y_test=np.where(Y_test==1.0,1,Y_test)
    
    X_test=test_list[i]
    X_test=X_test.loc[:,1:98]
    X_test=X_test.to_numpy()
    
    
    y_pred=model.predict(X_test)
    
    print(y_pred)
    print(Y_test)
    Accuracy=accuracy_score(Y_test,y_pred)
    Accuracy_list.append(Accuracy)


# In[62]:


print(Accuracy_list)


# In[1]:


fold1


# In[ ]:




