# main.py
# This is main process
from helper import list_item, resize, save_img, rename, showImage, huMonents, logarit
from setting import INPUT_PATH
import cv2
import math
import numpy as np
import os


for index,item_path in enumerate(list_item()):
    name = os.path.split(os.path.dirname(item_path))[1]
    print(">>>>>>", name)
    image = cv2.imread(item_path)
    # image = resize(origin)
    # image = showImage(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.Canny(image, 100,200)
    # ret,image = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    # hu = huMonents(image).flatten().tolist()
    # log = logarit(hu)
    # showImage(image)
    save_img("Color", "1",item_path, image)

train_data = np.array(train_data)
print(train_data.shape)