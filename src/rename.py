# main.py
# This is rename file programe
# Becarefull when use it

from helper import list_item, resize, save_img, rename
from setting import INPUT_PATH
import cv2
import time


for index,item in enumerate(list_item(INPUT_PATH)):
	# print(item)
    rename(item,"str",index)
    