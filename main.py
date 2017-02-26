# main.py
# This is main process
from helper import list_item, resize, save_img, rename, showImage
from setting import INPUT_PATH
import cv2


for index,item_path in enumerate(list_item(INPUT_PATH)):
    origin = cv2.imread(item_path)
    image = resize(origin)
    image = showImage(image)
    # showImage(image)
    save_img("Color", "1",item_path, image)
