import cv2
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
from helper import kmeans, resize, showImage

# name_list = ['cos', 'ant', 'coc', 'lot', 'str']
from SETTING import name_list,source_flowers_version_path,output_flowers_version_path,test_pic
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


for hoa_index, item in enumerate(listdir(source_flowers_version_path)):
    if item in name_list:
        parent_path = join(source_flowers_version_path, item)
        name_hoa = item
        output_parent_flowers_path = join(output_flowers_version_path, item)
        # print output_path
        if not exists(output_parent_flowers_path):
            makedirs(output_parent_flowers_path)
        index=0
        for item in listdir(parent_path):
            if not item.endswith(".png"):
                continue
            path_in = join(parent_path, item)
            index += 1
            # name = "1{}{:03}.jpg".format(name_list[name_hoa],index)
            name = item
            path_out = join(output_parent_flowers_path, name)
            print "Load:", path_in
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
                print "Save:", path_out
                cv2.imwrite(path_out, save)
            except Exception, err:
                print "Err", err, input
