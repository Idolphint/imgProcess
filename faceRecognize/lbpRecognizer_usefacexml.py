import cv2,os
import numpy as np

#casecading the xml file
cascade=cv2.CascadeClassifier("face.xml") #级联分类器

face_datas=[]
ids=[]
id=0
label_dict = []
#definig directory name where image data is stored
dir_name="E:\\2020-Spring\\IMGProcecss\\Face\\Face Recognition Data\\faces94\\female\\"
folder_name=os.listdir(dir_name)
for i in folder_name:
    #print(i) #这里是标签
    student_dir_name=dir_name+str(i)
    face_names=os.listdir(student_dir_name)
#creating blank list to save face_data and label
    
    for image_name in face_names:
    
        #creating image path
        image_path=student_dir_name+"\\"+image_name
    
        #reading image data in gray fromat
        face_data=cv2.imread(image_path,0)
        faces=cascade.detectMultiScale(face_data,1.5,5)
        for (x,y,w,h) in faces:
            #appending data in lists
            face_datas.append(face_data)#[y:y+h,x:x+w])
            hasId = False
            for idx, labi in enumerate(label_dict):
                if labi == i:
                    id = idx
                    ids.append(id)
                    hasId = True
                    break
            if not hasId:
                label_dict.append(i)
                id = len(label_dict) - 1
                ids.append(id)

face_datas = np.array(face_datas)
ids = np.array(ids)

idx_list = np.arange(len(face_datas))
#idx_list = np.array(range(len(images)))
np.random.shuffle(idx_list)
train_idx = idx_list[:int(0.8*len(idx_list))]
test_idx = idx_list[int(0.8*len(idx_list)) : ]

train_face = face_datas[train_idx]
train_Y = ids[train_idx]

test_face = face_datas[test_idx]
test_Y = ids[test_idx]

def LBPH_face():
    ############train and test
    recognizer=cv2.face.LBPHFaceRecognizer_create(radius = 2)

    #training the recognizer
    recognizer.train(train_face,train_Y)
    true_cnt = 0
    for idx, testi in enumerate(test_face):
        #predicting the label and confidence
        label,confidence=recognizer.predict(testi)
        
        if label == test_Y[idx]:
            true_cnt += 1
        else :
            print(confidence)
            print("should be: ", label_dict[label], "predict as :", label_dict[test_Y[idx]])
    print("finish test, acc: ", true_cnt/len(test_Y), "test num: ",len(test_Y))

def fisherFace():
    print("begin Fisher face")
    recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer.train(train_face, train_Y)
    true_cnt = 0
    for idx, testi in enumerate(test_face):
        #predicting the label and confidence
        label,confidence=recognizer.predict(testi)
        
        if label == test_Y[idx]:
            true_cnt += 1
        else :
            print(confidence)
            print("should be: ", label_dict[label], "predict as :", label_dict[test_Y[idx]])
    print("finish test, acc: ", true_cnt/len(test_Y), "test num: ",len(test_Y))

LBPH_face()