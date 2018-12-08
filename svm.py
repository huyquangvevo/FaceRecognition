import logging
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import os
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from joblib import dump,load
from sklearn.model_selection import KFold, cross_val_score
import pickle

rootPath = './Data'

n_classes = 5
id_peoples = np.arange(n_classes)

list_people = []

#n_samples,h,w = 44,64,64

y = np.array([],dtype=int)

print(id_peoples)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

isInited = False

a = np.array([0])
a = np.vstack((a,[1]))
a = np.vstack((a,[0]))

print(a)

for root, dirs, files in os.walk(rootPath):
    for index_dir,dir_ in enumerate(dirs):
        list_people.append(dir_)
        subDir = rootPath+"/"+dir_
        for f in os.listdir(subDir):
            y = np.append(y,index_dir)
            img = mpimg.imread(subDir+"/"+f)
            img = rgb2gray(img)
            img = img.ravel()
            img = np.append(img,index_dir)            
            if(isInited == False):
                print(img.shape)
                isInited = True
                data_people = img
            else:
                data_people = np.vstack((data_people,img))


                    
np.random.shuffle(data_people)

y = data_people[:,data_people.shape[1]-1]
y = y.astype(int)
data_people = data_people[:,:-1]

#save data 
dump(data_people,'./Estimated/datapeople.joblib')

n_samples,h,w = data_people.shape[0],64,64

X = data_people

n_components = 120
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)

X_train_pca = pca.transform(X)

target_names = list_people
dump(target_names,'./Estimated/name.joblib')


print("X train data shape ")
print(X_train_pca.shape)

clf = SVC(kernel='linear',C=0.1)
clf = clf.fit(X_train_pca,y)

dump(clf,'./Estimated/estimate.joblib')

