import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from helper import kmeans, resize

name_list = ['cos', 'ant', 'coc', 'lot', 'str']

for hoa_index, item in enumerate(listdir("./Hoa")[1:]):
    parent_path = join("./Hoa", item)
    name = item
    for index,item in enumerate(listdir(parent_path)[1:]):
        path_in = join(parent_path, item)
        name = "1{}{:03}.jpg".format(name_list[hoa_index],index)
        path_out = join(parent_path, "Ok", name)
        print path_out
        try:
            image = cv2.imread(path_in)
            image = resize(image)
            image = kmeans(image,2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = cv2.threshold(frame, frame, 20, 255, THRESH_BINARY)
            ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
            # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            cv2.imwrite(path_out, image)
            # images = [image, th1, th2, th3]
            # cv2.imshow("OK", th1)
        except Exception, err:
            print "Err", err, input

# image = cv2.imread("/Users/marsch/Projects/IPAI/FlowerDetect/tools/Hoa/Hoa Canh Buom/Screen Shot 2017-01-10 at 20.21.32.png")
# image = kmeans.kmean(image,2)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imgshow("Image", image)
