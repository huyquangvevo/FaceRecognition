from joblib import load,dump
import pickle
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import tkinter as tk
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from PIL import Image,ImageTk
import os


gui = Tk()
gui.title("Face Recognition")


filename = askopenfilename()
image_ = Image.open(filename)
photo = ImageTk.PhotoImage(image_)




cv = tk.Canvas(width=400,height=400)
cv.pack(side='top',fill='both',expand='yes')
cv.create_image(200,100,image=photo,anchor='center')

id_images=os.path.basename(filename)
id_images = os.path.splitext(id_images)[0][:-1]
text_true = "True: " + id_images


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread(filename)
img = rgb2gray(img)
img = img.ravel()

X = load('./Estimated/datapeople.joblib')
X = np.vstack((X,img))

n_components = 120
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)

X = pca.transform(X)
img = X[-1:,:]

clf = load('./Estimated/estimate.joblib')


name_people = load('./Estimated/name.joblib')


y = clf.predict(img)

result = name_people[y[0]]

#gui.mainloop()
print(result)


result = "Predicted: " + result

cv.create_text(200,200, fill="darkblue",font=("Times 20 italic bold",14),anchor='center',
                        text=result)

cv.create_text(200,250,fill="darkblue",font=("Times 20 italic bold",16),anchor='center',
                        text=text_true)

gui.mainloop()
