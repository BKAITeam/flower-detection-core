import cv2
import numpy as np

def resize(image):
    r = 100.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 100)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image


def kmeans(img,k, loop):
    # img = cv2.imread('1ant009.jpg')
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, loop, 1.0)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, loop, 1.0)
    ret,label,center = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # cv2.imwrite("ok3.jpg", res2)
    return label,center,res2


def onClusterChange(cluster):
    global num_of_cluster
    num_of_cluster = cluster
    global update
    update = True
    return None

def onKmeanChange(loop):
    global loop_of_kmean
    loop_of_kmean = loop
    global update
    update = True

def onErosionChange(loop):
    global Erosion
    Erosion = loop
    global update
    update = True

def onDilasionChange(loop):
    global Dilasion
    Dilasion = loop
    global update
    update = True

def fill(board,image,m,n):
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            board[m*100+i][n*image.shape[1]+j] = image[i][j]
    return board

def joinMutipleImage(*images):
    m = int(round(len(images)/2.0))
    n = 2
    width = 0
    height = 200
    for i in xrange(m):
        width = width + images[i].shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # for item in images:
    k = 0
    for i in xrange(n):
        for j in xrange(m):
            if k< len(images):
                canvas = fill(canvas,images[k],i,j)
                k = k + 1
    return canvas

def extraction(image,n, count):
    result = []
    label, center, kmean_image = kmeans(image,n, count)
    # print "KMEAN IMAGE",type(image_kmean),"IMAGE",type(image)
    # new_image = image.tolist()
    # print "NEW IMAGE",type(new_image)

    # new_image = map(lambda row: map(lambda point: point if point != center[0].tolist() else [0,0,0], row), new_image)
    # print "RESULT IMAGE",type(result)
    # new_image = np.array(new_image, dtype=np.uint8)
    # result.append(new_image)
    for i in xrange(n):
        new_image = image.tolist()
        kmean_image_list = kmean_image.tolist()
        # new_image = map(lambda row: map(lambda point: point if point != center[i].tolist() else [0,0,0], row), kmean_image_list)
        for index_row,row in enumerate(kmean_image_list):
            for index_column,point in enumerate(row):
                if point != center[i].tolist():
                    new_image[index_row][index_column]=[0,0,0]

        new_image = np.array(new_image, dtype=np.uint8)
        # new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        # ret,new_image = cv2.threshold(new_image,0,255,cv2.THRESH_OTSU)
        result.append(new_image)
    return result

def showImage(image):
    global num_of_cluster
    num_of_cluster = 3
    global loop_of_kmean
    loop_of_kmean = 5
    global Erosion
    Erosion = 0
    global Dilasion
    Dilasion = 0
    image = resize(image)

    # print image
    wname = "Image"
    cv2.namedWindow(wname)

    # create trackbars for color change
    cv2.createTrackbar('Cluster',wname,3,20,onClusterChange)
    cv2.createTrackbar('KCount',wname,5,20,onKmeanChange)
    # cv2.createTrackbar('Erosion',wname,0,1,onErosionChange)
    # cv2.createTrackbar('Dilasion',wname,0,1,onDilasionChange)
    global update
    update = True
    run = True
    while(run):
        if update:
            images = extraction(image,num_of_cluster, loop_of_kmean)
            show_image = joinMutipleImage(*images)
            cv2.imshow("Image", show_image)
            update = False

        key = cv2.waitKey(0)
        if key > ord('0') and key <= ord('9'):
            return images[key-48-1]
        elif key == ord('q'):
            run = False

def huMonents(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(image))
