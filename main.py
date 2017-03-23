# main.py
# This is main process
from helper import list_item, resize, save_img, rename, showImage, huMonents, logarit
from setting import INPUT_PATH
import cv2
import math
import numpy as np
import os


train_data = []

def output_layer(n):
    arr = np.zeros((10), dtype=np.float32)
    arr[n] = 1.0
    return arr

one = output_layer(1)
two = output_layer(2)
three = output_layer(3)

label = {
    "Hoa Canh Buom": output_layer(0),
    "Hoa Hong": output_layer(1),
    "Hoa Hong Mon": output_layer(2),
    "Hoa Huong Duong": output_layer(3),
    "Hoa Ly": output_layer(4),
    "Hoa Mao Ga": output_layer(5),
    "Hoa Sen": output_layer(6),
    "Hoa Thien Dieu": output_layer(7),
    "Hoa Thuoc Duoc": output_layer(8),
    "Hoa Trang": output_layer(9)
}

for index,item_path in enumerate(list_item()):
    name = os.path.split(os.path.dirname(item_path))[1]
    print ">>>>>>", label[name], name
    origin = cv2.imread(item_path)
    image = resize(origin)
    # image = showImage(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.Canny(image, 100,200)
    # ret,image = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    hu = huMonents(image).flatten().tolist()
    log = logarit(hu)
    # print hu
    # print log
    train_data.append(log)
    # showImage(image)
    # save_img("Edge", "4",item_path, image)

train_data = np.array(train_data)
print(train_data.shape)