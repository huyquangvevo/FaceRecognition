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

rootPath = './Data'

n_classes = 5
# gán id cho mỗi người 
id_peoples = np.arange(n_classes)

# Tên đối tượng
list_people = []

#n_samples,h,w = 44,64,64
# Tạo mảng nhãn empty
y = np.array([],dtype=int)

print(id_peoples)

# chuyển từ ảnh màu sang ảnh đa mức xám
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

isInited = False

a = np.array([0])
a = np.vstack((a,[1]))
a = np.vstack((a,[0]))

print(a)
# đọc các thư mục con, và file nằm trong thư mục con đó 
for root, dirs, files in os.walk(rootPath):
    for index_dir,dir_ in enumerate(dirs):
        list_people.append(dir_) # thêm tên thư mục vào list, tên thư mục là tên người
        subDir = rootPath+"/"+dir_ # path đến thư mục con
        for f in os.listdir(subDir):
            y = np.append(y,index_dir) # thêm id của người 
            img = mpimg.imread(subDir+"/"+f) # đọc ảnh 
            img = rgb2gray(img) # chuyển sang ảnh đa mức xám
            img = img.ravel()  # tạo ma trận dữ liệu điểm ảnh
            img = np.append(img,index_dir)   # thêm id của người vào cuối ma trận dư liệu          
            if(isInited == False): # nếu là lần đầu thì khởi tạo biến data_people
                print(img.shape)
                isInited = True
                data_people = img
            else:
                data_people = np.vstack((data_people,img)) # thêm dữ liệu mỗi ảnh và data 


                    
np.random.shuffle(data_people) # trộn mảng 

y = data_people[:,data_people.shape[1]-1] # lấy id của người ứng với mỗi vector 
y = y.astype(int) # chuyển sang kiểu int
data_people = data_people[:,:-1] # bỏ đi cột cuối của ma trận, (bỏ id)

#save data  , lưu tập dữ liệu 
dump(data_people,'./Estimated/datapeople.joblib')

n_samples,h,w = data_people.shape[0],64,64 # số mẫu

X = data_people 

n_components = 120 # số thành phần PCA giữ lại 
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X) # train PCA
dump(pca,'./Estimated/pca.joblib')

X_train_pca = pca.transform(X) # transform X


target_names = list_people
dump(target_names,'./Estimated/name.joblib') # lưu tên người ra file 


print("X train data shape ")
print(X_train_pca.shape)

clf = SVC(kernel='linear',C=0.1) # gọi SVM(SVC)
clf = clf.fit(X_train_pca,y) # học

dump(clf,'./Estimated/estimate.joblib') # lưu tham số tìm đc(w,b)