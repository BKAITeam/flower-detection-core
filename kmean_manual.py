import cv2
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
from helper import kmeans, resize, showImage

name_list = ['cos', 'ant', 'coc', 'lot', 'str']

def doing():
    pass

def process(image):
    wname = "Image"
    cv2.namedWindow(wname)

    # create trackbars for color change
    cv2.createTrackbar('K:',wname,1,10,doing)
    cv2.createTrackbar('L:',wname,1,10,doing)

    cv2.imshow("Image", image)
    key = cv2.waitKey(0)


for hoa_index, item in enumerate(listdir("./Hoa")[1:]):
    parent_path = join("./Hoa", item)
    name = item
    path_ok = join(parent_path, "Ok3")
    if not exists(path_ok):
        makedirs(path_ok)
    index=0
    for item in listdir(parent_path)[1:]:
        if not item.endswith(".png"):
            continue
        path_in = join(parent_path, item)
        index += 1
        name = "1{}{:03}.jpg".format(name_list[hoa_index],index)
        path_out = join(path_ok, name)
        # if not os.path.exists(path_out):
            # os.makedirs(path_out)
        print path_out
        try:
            image = cv2.imread(path_in)
            # Begin processing
            # process(image)
            save = showImage(image)
            # image = resize(image)
            # label,center,image = kmeans(image,2)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
            # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,7)
            # End processing
            cv2.imwrite("ok.png", save)
        except Exception, err:
            print "Err", err, input
