# main.py
# This is rename file programe
# Becarefull when use it
from helper import list_item, resize, save_img, rename
from setting import INPUT_PATH
import cv2


for index,item in enumerate(list_item(INPUT_PATH)):
    rename(item,"str",index)
