import cv2
import random
import math
import numpy as np


def distance(a,b):
    d = math.sqrt((a[1]-b[1])**2 + (a[2]-b[2])**2)
    return d


def center_min(point, center):
    min = 10000.0
    result = 0
    for i in xrange(len(center)):
        d = distance(point, center[i])
        if min >= d:
            min = d
            result = i
    return result


def kmeans(img, cluster, count):
    # image = cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
    image = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    h, w, d  = image.shape
    center = []
    while(center.__len__()!=cluster):
        point = (random.randint(0,h-1),random.randint(0,w-1))
        if point not in center:
            center.append(point)
    center = [(50,50),(75,75),(90,90)]
    point_of_cluster = [[] for i in xrange(cluster)]
    center = map(lambda x: image[x[0]][x[1]].tolist(), center)

    label = np.zeros((h,w), dtype=np.uint8)

    for k in xrange(count):
        point_of_cluster = [[] for i in xrange(cluster)]
        print len(point_of_cluster)
        for i in xrange(h):
            for j in xrange(w):
                label[i][j] = center_min(image[i][j], center)
                point = (i,j,distance(image[i][j], center[label[i][j]]))
                point_of_cluster[label[i][j]].append(point)

        for i in xrange(cluster):
            if len(point_of_cluster[i])>0:
                point_of_cluster[i] = sorted(point_of_cluster[i], key=lambda x: x[2])
                middle_point = len(point_of_cluster[i])/2
                point = point_of_cluster[i][middle_point]
                # print point_of_cluster[i]
                center[i] = image[point[0]][point[1]]
    return label


def show(cluster, count):
    origin = cv2.imread("source1.png")
    label = kmeans(origin, cluster, count)
    h, w = label.shape
    results = np.ones((cluster,h,w,3), dtype=np.uint8)
    for k in xrange(cluster):
        for i in xrange(h):
            for j in xrange(w):
                if label[i][j] == k:
                    results[k][i][j] = origin[i][j]
    origin = cv2.cvtColor(origin,cv2.COLOR_BGR2LAB)
    cv2.imshow("Image",origin)
    cv2.imshow("Image 1",results[0])
    cv2.imshow("Image 2",results[1])
    cv2.imshow("Image 3",results[2])
    # cv2.imshow("Image 4",results[3])
    # cv2.imshow("Image 5",results[4])
    cv2.waitKey(0)

show(3,5)
