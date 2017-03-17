# main.py
# This is main process
from helper import list_item, resize, save_img, rename, showImage, huMonents, logarit
from setting import INPUT_PATH
import cv2
import math

print INPUT_PATH

for index,item_path in enumerate(list_item(INPUT_PATH)):
    print ">>>>>>", item_path
    origin = cv2.imread(item_path)
    image = resize(origin)
    # image = showImage(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.Canny(image, 100,200)
    # ret,image = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    hu = huMonents(image).flatten().tolist()
    log = logarit(hu)
    # print hu
    print log
    # showImage(image)
    # save_img("Edge", "4",item_path, image)
