import cv2
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
from helper import kmeans, resize, showImage
from setting import NAME_LIST, INPUT_PATH, OUT_PATH


for hoa_index, item in enumerate(listdir(INPUT_PATH)):
    if item in NAME_LIST:
        parent_path = join(INPUT_PATH, item)
        name_hoa = item
        OUT_PATH = join(OUT_PATH, item)
        if not exists(OUT_PATH):
            makedirs(OUT_PATH)
        index=0
        for item in listdir(parent_path):
            if not item.endswith(".png"):
                continue
            path_in = join(parent_path, item)
            index += 1
            # name = "1{}{:03}.jpg".format(name_list[name_hoa],index)
            name = item
            path_out = join(OUT_PATH, name)
            print("Load:", path_in)
            try:
                image = cv2.imread(path_in)
                save = showImage(image)
                # image = resize(image)
                # label,center,image = kmeans(image,2)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
                # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
                # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,7)
                # End processing
                print("Save:", path_out)
                cv2.imwrite(path_out, save)
            except Exception as err:
                print("Err", err, input)
