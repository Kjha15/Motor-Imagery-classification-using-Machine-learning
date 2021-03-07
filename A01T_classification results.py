############################################################################
# This code is a contribution towards the term project
# of the Neural machine interface (ENGR-845) at San Francisco State University
#Authors of this code are Kirtan Amitkumar Jha and Naren VIshwanath Kalburgi 
#Authors would like to give credit to the python dependencies documentations 
#through which thet learnt how to build the classifiers and what are the para
#-meters to be kept for different machine learning algorithms
###########################################################################


# In[50]: Importing python dependencis 



import matplotlib 
import pandas as pd
import numpy as np
import math
import re

import os
import sklearn
import datetime
from matplotlib import pyplot as plt

from sklearn import model_selection, svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, average_precision_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline

#BCI MNE Imports
import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne import Epochs, pick_types, events_from_annotations


# In[51]: Importing raw files into MNE python



raw_files = [read_raw_edf('C:\\Users\\harsh\\Documents\\MATLAB\\A01T.gdf', preload=True, stim_channel='auto')]


# In[52]:



raw_files[0]


# In[53]:


subject_list=[]
i=0
while True:
    if(i == 594):
        break
    a=raw_files[0+i:6+i]  
    subject_list.append(a)
    i=i+6


# In[54]:


raw=raw_files[0]


# In[55]: Remove 3 raw ECOG channels 


raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')


# In[79]: Add events id 


events, _ = events_from_annotations(raw)
#here there was the main challenge to find the exact event ids because according to annotations its 768 and above but due to MNE version problem it was different
#So instead of 288 trials, 144 (half) trials were observed and trained and tested
#Output of 21st cell has '144 matching events found' which satisfies the above comment
event_id = dict(left=7, right=8,foot=9, tongue=10)
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')


# In[80]:


tmin, tmax = 1., 4.
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)


# In[81]: Plot epochs average


#left hand epoch average plot
evoked = epochs['left'].average()
print(evoked)
evoked.plot(time_unit='s')
#right hand epoch average plot
evoked = epochs['right'].average()
print(evoked)
evoked.plot(time_unit='s')

# Foot epoch average plot

evoked = epochs['foot'].average()
print(evoked)
evoked.plot(time_unit='s')

# Tongue epoch average plot

evoked = epochs['tongue'].average()
print(evoked)
evoked.plot(time_unit='s')
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
epochs_train


# In[82]:


labels = epochs.events[:,-1] - 769 + 1

data = epochs.get_data()


# In[83]: Wavelet packet decomposition 


import pywt

# signal is decomposed to level 5 with 'db4' wavelet

def wpd(X): 
    coeffs = pywt.WaveletPacket(X,'db4',mode='symmetric',maxlevel=5)
    return coeffs
             
def feature_bands(x):
    
    Bands = np.empty((8,x.shape[0],x.shape[1],30)) # 8 freq band coefficients are chosen from the range 4-32Hz
    
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
             pos = []
             C = wpd(x[i,ii,:]) 
             pos = np.append(pos,[node.path for node in C.get_level(5, 'natural')])
             for b in range(1,9):
                 Bands[b-1,i,ii,:] = C[pos[b]].data
        
    return Bands

wpd_data = feature_bands(data)


# In[84]: 10 K-cross fold validation with CSP 


from mne.decoding import CSP # Common Spatial Pattern Filtering
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from sklearn.model_selection import ShuffleSplit

# OneHotEncoding Labels
enc = OneHotEncoder()
X_out = enc.fit_transform(labels.reshape(-1,1)).toarray()

# Cross Validation Split and splitting data into training and testing data
cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc = []
ka = []
prec = []
recall = []


# In[87]: Start building the classifiers 


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 32, 
                         kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
    classifier.add(Dropout(p = 0.5))
    for itr in range(1):
        classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'tanh', 
                             kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
        classifier.add(Dropout(p = 0.5))    
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier


# In[88]:


for train_idx, test_idx in cv.split(labels):
    
    Csp = [];ss = [];nn = [] # empty lists
    
    label_train, label_test = labels[train_idx], labels[test_idx]
    y_train, y_test = X_out[train_idx], X_out[test_idx]
    
    # CSP filter applied separately for all Frequency band coefficients
    
    Csp = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for _ in range(8)]
    ss = preprocessing.StandardScaler()

    X_train = ss.fit_transform(np.concatenate(tuple(Csp[x].fit_transform(wpd_data[x,train_idx,:,:],label_train) for x  in range(8)),axis=-1))

    X_test = ss.transform(np.concatenate(tuple(Csp[x].transform(wpd_data[x,test_idx,:,:]) for x  in range(8)),axis=-1))
    
    nn = build_classifier()  
    
    nn.fit(X_train, y_train, batch_size = 32, epochs = 300)
    
    y_pred = nn.predict(X_test)
    pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)

    acc.append(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    ka.append(cohen_kappa_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    prec.append(precision_score(y_test.argmax(axis=1), pred.argmax(axis=1),average='weighted'))
    recall.append(recall_score(y_test.argmax(axis=1), pred.argmax(axis=1),average='weighted'))


# In[90]: Take out average of K-cross validation and see the precision and kappa values


import pandas as pd

scores = {'Accuracy':acc,'Kappa':ka,'Precision':prec,'Recall':recall}

Es = pd.DataFrame(scores)

avg = {'Accuracy':[np.mean(acc)],'Kappa':[np.mean(ka)],'Precision':[np.mean(prec)],'Recall':[np.mean(recall)]}

Avg = pd.DataFrame(avg)


K = pd.concat([Es,Avg])

K.index = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','Avg']
K.index.rename('Fold',inplace=True)

print(K)


# In[91]: Linear Discrimanant Analysis


def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    return accuracy_score(y_true, y_pred)


# In[92]:


scores = [] #empty list
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
#80% tarining and 20% testing data
cv = ShuffleSplit(5, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)


# In[93]:


LDA = LinearDiscriminantAnalysis()
csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
clf = Pipeline([('CSP', csp), ('LDA', LDA)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1, scoring=make_scorer(classification_report_with_accuracy_score))


# In[68]:


class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))


# In[69]:


LDA = LinearDiscriminantAnalysis()
csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
clf = Pipeline([('CSP', csp), ('LDA', LDA)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1, scoring=make_scorer(classification_report_with_accuracy_score))


# In[95]:


class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
lda=(np.mean(scores))*100
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))


# In[96]: print classification accuracy


lda


# In[97]: K-Nearest Neighbours


from sklearn.neighbors import KNeighborsClassifier

KNN=KNeighborsClassifier(n_neighbors=2)
clf = Pipeline([('CSP', csp), ('KNN', KNN)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1, scoring=make_scorer(classification_report_with_accuracy_score))


# In[98]:


class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
knn=np.mean(scores)*100
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))


# In[99]: print classification accuracy 


print(knn)


# In[100]: Multi layer perceptron 


from sklearn.neural_network import MLPClassifier
MLP=MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
clf = Pipeline([('CSP', csp), ('MLP', MLP)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1, scoring=make_scorer(classification_report_with_accuracy_score))


# In[101]:


class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
mlp=np.mean(scores)*100
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))


# In[102]: Support Vector machine 


from sklearn import svm
SVM=svm.SVC()
csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
clf = Pipeline([('CSP', csp), ('SVM', SVM)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1, scoring=make_scorer(classification_report_with_accuracy_score))


# In[103]:


class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
svm=np.mean(scores)*100
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))

