from unittest import result
import cv2
from tensorflow import keras
import json
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import shutil


config_path = "../config/config.json"
cfg = json.load(open(config_path,'r'))
test_path = cfg['data_dir']['test']
infer_model_path = cfg['infer_params']['infer_model_path']
test_img = []
test_imgs = []
y_gt = []
img_names = []
folders = glob.glob(test_path + '/' + '*')
i = 0
for folder in folders:
    imgs = glob.glob(folder + '/' + '*')
    for img in imgs:
        i+=1
        gt_folder = folder.split('/')[-1]
        if gt_folder=='PNEUMONIA':
            y_gt.append(1)
        else:
            y_gt.append(0)
        test_img = cv2.imread(img)
        img_names.append(img)
        test_img_resized = cv2.resize(test_img, (299,299)) 
        test_imgs.append(test_img_resized)


test_imgs = np.array(test_imgs)
test_imgs = test_imgs/255.0

model = keras.models.load_model(infer_model_path)


y_prediction = model.predict(test_imgs)
y_probas = y_prediction
y_prediction = np.argmax(y_prediction, axis=1).tolist()


result = confusion_matrix(y_gt, y_prediction)
tn, fp, fn, tp = result.ravel()


print(tp,fp,tn,fn)

print("precision is:", (tp)/(tp+fp))
print("recall is:", (tp)/(tp+fn))

print(result)






