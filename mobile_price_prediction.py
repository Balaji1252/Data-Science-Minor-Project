#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt #import the required libraries
import seaborn as sns


# In[2]:


train=pd.read_csv(r'mobile_price_range_data.csv') #store the training dataset in a variable(train)
test=pd.read_csv(r'test.csv.xls') #store the testing dataset in a variable(test)


# In[3]:


pd.set_option('display.max_row',None)
pd.set_option('display.max_columns',None) #display all the number of rows and cols


# In[4]:


train.head() #display the top 5 records of the train dataset(rows)


# In[5]:


test.head() #display the top 5 records of the test dataset(rows)


# In[6]:


test.drop('id',axis=1,inplace=True) #to drop the id from test dataset


# In[7]:


test.head()


# In[8]:


sns.countplot(train['price_range']) # to display and check for the imbalance of the dataset, hence all are 500 rows which indicates the dataset is balance or not


# In[9]:


train.shape,test.shape #to check the no of rows nd cols


# In[10]:


train.isnull().sum() # to check whether the training data set contains null values or not


# In[11]:


train.info() #to check the information of the dataset


# In[12]:


test.info() #to check the information of testing dataset


# In[13]:


train.describe() #describe functions is used to display the count,mean,std deviation,min,25%,50%,75%,max


# In[14]:


train.plot(x='price_range',y='ram',kind='scatter')
plt.show() # apply some visualisation of the data set


# In[15]:


train.plot(x='price_range',y='battery_power',kind='scatter')
plt.show()


# In[16]:


train.plot(x='price_range',y='fc',kind='scatter')
plt.show()


# In[17]:


train.plot(x='price_range',y='n_cores',kind='scatter')
plt.show()


# In[18]:


import seaborn as sns
plt.figure(figsize=(20,20))
sns.heatmap(train.corr(),annot=True,cmap=plt.cm.Accent_r)
plt.show()  #to check the correlation of the dataset(not mandate)


# In[19]:


train.plot(kind='box',figsize=(20,10)) #to check the dataset contains outliers(abnormal distance from other values) or not(here it not contains)


# In[20]:


X=train.drop('price_range',axis=1)
Y=train['price_range'] #to split the data set from independent to dependent features(heading)


# In[22]:


from sklearn.model_selection import train_test_split #split the data set according to dependent,independent,test size and random state
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.1,random_state=101)


# In[24]:


from sklearn.preprocessing import StandardScaler #apply the standardisation on all the training and testing data set
#Standardisation is done here(makes all the feature value in a particular range 0-1)
sc=StandardScaler() # storing it in variable sc
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
test=sc.transform(test)


# In[25]:


X_train


# In[26]:


X_test


# In[27]:


test


# In[28]:


from sklearn.tree import DecisionTreeClassifier #importing decision tree classifier 
dtc=DecisionTreeClassifier() # store it in a variable
dtc.fit(X_train, Y_train) # train using Xtrain, Y_train


# In[29]:


pred=dtc.predict(X_test)
pred #Test the model using x_test data set


# In[30]:


from sklearn.metrics import accuracy_score, confusion_matrix # to check the accuracy and to generate confuison matrix
dtc_acc= accuracy_score(pred,Y_test)
print(dtc_acc) #to print the accuracy 
print(confusion_matrix(pred,Y_test)) # to print the confusion matric of the decision tree classifier


# In[31]:


from sklearn.svm import SVC #to import SVC(support vector classifier) for knn algo
knn=SVC()
knn.fit(X_train,Y_train)


# In[32]:


pred1=knn.predict(X_test) #test the model
pred1


# In[34]:


from sklearn.metrics import accuracy_score #to check the accuray
svc_acc=accuracy_score(pred1,Y_test) #store in another variable
print(svc_acc) #print the accuracy
print(confusion_matrix(pred1,Y_test)) #to print the confusion matrix


# In[36]:


from sklearn.linear_model import LogisticRegression #it is a classification(result in real numbers)
lr=LogisticRegression()
lr.fit(X_train,Y_train) #train with xtrain,ytrain


# In[37]:


pred2=lr.predict(X_test)
pred2 #test the model


# In[38]:


from sklearn.metrics import accuracy_score #to check the accuray
lr_acc=accuracy_score(pred2,Y_test) #store in another variable
print(lr_acc) #print the accuracy
print(confusion_matrix(pred2,Y_test)) #to print the confusion matrix
#Here out of the three models, decision tree classifier, knn(svc), logistic regression , logistic regression model has the highest accuracy


# In[39]:


plt.bar(x=['dtc','svc','lr'],height=[dtc_acc,svc_acc,lr_acc])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.show()  #to visualise in bar chart (algos and its accuracy)


# In[40]:


lr.predict(test) #testing the log reg with sep testing file


# In[41]:


#This is the output, 0-Low cost, 1-Medium cost, 2-High cost, 3-Very high cost. Out of the three alogs dtc,svc and lr , lr has the best accuracy


# In[ ]:




