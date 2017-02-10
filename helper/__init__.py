import cv2
import numpy as np

def resize(image):
    r = 100.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 100)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image


def kmeans(img,k):
    # img = cv2.imread('1ant009.jpg')
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # cv2.imwrite("ok3.jpg", res2)
    return res2
