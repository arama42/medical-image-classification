from unittest import result
import cv2
from tensorflow import keras
import json
import glob
import numpy as np
from sklearn.metrics import confusion_matrix


config_path = "../config/config.json"
cfg = json.load(open(config_path,'r'))
test_path = cfg['data_dir']['test']
test_img = []
test_imgs = []
y_gt = []
folders = glob.glob(test_path + '/' + '*')
for folder in folders:
    imgs = glob.glob(folder + '/' + '*')
    for img in imgs:
        gt_folder = folder.split('/')[-1]
        if gt_folder=='PNEUMONIA':
            y_gt.append(1)
        else:
            y_gt.append(0)
        test_img = cv2.imread(img)
        test_img_resized = cv2.resize(test_img, (224,224))
        test_imgs.append(test_img_resized)


test_imgs = np.array(test_imgs)
test_imgs = test_imgs/255.0


# test_data=test_datagen.flow_from_directory(self.test_path,
#                                                 target_size=(self.image_shape,self.image_shape),
#                                                 batch_size=self.batch_size,
#                                                 class_mode='categorical')

model = keras.models.load_model('../../Dataset/Models/baseline_resnet_chest_xray_e10_b64.h5')
y_prediction = model.predict(test_imgs)
y_prediction = np.argmax(y_prediction, axis=1).tolist()

print(y_prediction)
input("y_prediction")

print(y_gt)


result = confusion_matrix(y_gt, y_prediction)
tn, fp, fn, tp = result.ravel()


print(tp,fp,tn,fn)

print("precision is:", (tp)/(tp+fp))
print("recall is:", (tp)/(tp+fn))
