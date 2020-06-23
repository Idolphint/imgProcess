import cv2, os
from numpy import *
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC

CNT = 0
cascadeLocation = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadeLocation)


#test_i = cv2.imread("E:/2020-Spring/IMGProcecss/Face/Face-Recogntion/female/ksunth.18.jpg", 0)
#better_img(test_i)
def prepare_dataset(images, labels, directory, lab, CNT):
	paths = [filename for filename in os.listdir(directory)]
	value_cnt=0
	row = 192
	col = 192
	for image_path in paths:
		image_pil = cv2.imread(os.path.join(directory, image_path), 0)
		#image_pil = Image.open(image_path).convert('L')
		#image_pil = cv2.equalizeHist(image_pil)
		image_pil = 255. * (image_pil - np.amin(image_pil)) / (np.amax(image_pil) - np.amin(image_pil))
		image = np.resize(image_pil, (row, col)).astype(np.int8)
		
		images[CNT] = image
		labels[CNT] = lab
		CNT+=1
	return images,labels, row, col, CNT

images = np.zeros((799, 192, 192), dtype=np.int)
labels = np.zeros((799))
directory = 'E:/2020-Spring/IMGProcecss/Face/Face-Recogntion/female/'
images, labels, row, col, CNT = prepare_dataset(images, labels, directory, 0, CNT) #female

directory = 'E:/2020-Spring/IMGProcecss/Face/Face-Recogntion/malestaff/'
images, labels, row, col, CNT = prepare_dataset(images, labels, directory, 1, CNT) #female

idx_list = np.arange(images.shape[0])
#idx_list = np.array(range(len(images)))
np.random.shuffle(idx_list)
train_idx = idx_list[:int(0.9*len(idx_list))]
test_idx = idx_list[int(0.9*len(idx_list)) : ]

n_components = 10

train_img = images[train_idx]
print(train_img.dtype)
train_img = train_img.astype(np.int32)
train_lab = np.array(np.array(labels)[train_idx])

test_img = np.array(np.array(images)[test_idx])
test_img = test_img.astype(int)
test_lab = np.array(np.array(labels)[test_idx])
pca = PCA(n_components=n_components)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf'),param_grid)

training_data = []
for i in range(len(train_img)):
	training_data.append(train_img[i].flatten())
pca = pca.fit(training_data)

transformed = pca.transform(training_data)
clf.fit(transformed,train_lab)
######train finish 

def get_key(dict_, value):
    for k, v in dict_.items():
        if v == value:
            return k
ri = 0
for i in range(len(test_img)):
	img = test_img[i]
	X_test = pca.transform(img.reshape((1,-1)))
	Y_test = test_lab[i]
	mynbr = clf.predict(X_test)
	if mynbr[0] == Y_test:
		ri += 1
	
print("test all ", len(test_img), "acc: ", ri/len(test_img), ri)