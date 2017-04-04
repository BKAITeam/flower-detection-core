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
    origin = cv2.imread(item_path)
    image = resize(origin)
    save_img("Resized", "1",item_path, image)
