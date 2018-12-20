import logging
import matplotlib.pyplot as plt

import os
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from joblib import dump,load

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
#print(data_people[0])

y = data_people[:,data_people.shape[1]-1]
y = y.astype(int)
data_people = data_people[:,:-1]

#save data 
#dump(data_people,'datapeople.joblib')
print(data_people.shape)
#print(y)

n_samples,h,w = data_people.shape[0],64,64


# prediction

target_names = list_people

X = data_people

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)

n_components = 120

pca = PCA(n_components=n_components,svd_solver='randomized',whiten=True).fit(X_train)

print(pca.components_.shape)

eigenfaces = pca.components_.reshape((n_components,h,w))


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#X_train_pca = X_train
#X_test_pca = X_test

print("X train data shape ")
print(X_train_pca.shape)
'''
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
'''
clf = SVC(kernel='linear',C=0.1)

clf = clf.fit(X_train_pca, y_train)
#print("Best estimator found by grid search: ")
#print(clf.best_estimator_)

y_pred = clf.predict(X_test_pca)

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))




def plot_gallery(images, titles, h, w, n_row=4, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

#dump
#dump(clf,'people.joblib')

plt.show()


#dump


# Predict a image
'''
predImg = mpimg.imread(rootPath+'/4.jpg')

predImg = rgb2gray(predImg)
predImg = predImg.ravel()

predImg = np.vstack((predImg,predImg))
predImg = predImg[:-1,:]
#print ("Shape: ")
#print(predImg.shape)

print("Predict Img : ")
pcaImg = pca.transform(predImg)
result = clf.predict(pcaImg)
print(list_people[result[0]])
'''

    

