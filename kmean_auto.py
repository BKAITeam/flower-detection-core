import cv2
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
from helper import kmeans, resize

# name_list = ['cos', 'ant', 'coc', 'lot', 'str']
name_list = {
    "Hoa Canh Buom": 'cos',
    "Hoa Hong Mon": "ant",
    "Hoa Mao Ga": "coc",
    "Hoa Sen": "lot",
    "Hoa Thien Dieu": "str"
}
for hoa_index, item in enumerate(listdir("./Hoa")):
    parent_path = join("./Hoa", item)
    name_hoa = item
    if item != "Hoa Thien Dieu":
        continue
    path_ok = join(parent_path, "Ok2")
    if not exists(path_ok):
        makedirs(path_ok)
    index=0
    for item in listdir(parent_path)[1:]:
        if not item.endswith(".png"):
            continue
        path_in = join(parent_path, item)
        index += 1
        name = "0{}{:03}.png".format(name_list[name_hoa],index)
        path_out = join(path_ok, name)
        # if not os.path.exists(path_out):
            # os.makedirs(path_out)
        print path_out
        try:
            image = cv2.imread(path_in)
            image = resize(image)
            label,center, image = kmeans(image,3,5)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret,image = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
            # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,7)
            print path_out
            cv2.imwrite(path_out, image)
        except Exception, err:
            print "Err", err, input
