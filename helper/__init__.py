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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, loop, 1.0)
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
    print "Num of cluster", cluster
    return None

def onKmeanChange(loop):
    global loop_of_kmean
    loop_of_kmean = loop
    global update
    update = True
    print "Num of Kmean", loop

def fill(board,image,m,n):
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            board[m*100+i][n*image.shape[1]+j] = image[i][j]
    return board

def joinMutipleImage(*images):
    print "JOIN", len(images)
    m = int(round(len(images)/2.0))
    n = 2
    width = 0
    height = 200
    print "Width will equal 0", width
    for i in xrange(m):
        print i, images[i].shape[1]
        width = width + images[i].shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    print canvas.shape, width
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
    label, center, image = kmeans(image,n, count)
    for i in xrange(n):
        new_image = image.tolist()
        new_image = map(lambda row: map(lambda point: point if point == center[i].tolist() else [0,0,0], row), new_image)
        new_image = np.array(new_image, dtype=np.uint8)
        result.append(new_image)
    return result

def showImage(image):
    global num_of_cluster
    num_of_cluster = 3
    global loop_of_kmean
    loop_of_kmean = 5
    image = resize(image)

    # print image
    wname = "Image"
    cv2.namedWindow(wname)

    # create trackbars for color change
    cv2.createTrackbar('Cluster',wname,3,20,onClusterChange)
    cv2.createTrackbar('KCount',wname,1,10,onKmeanChange)
    global update
    update = True
    run = True
    while(run):
        print "Running ...", update, num_of_cluster
        if update:
            images = extraction(image,num_of_cluster, loop_of_kmean)
            show_image = joinMutipleImage(*images)
            cv2.imshow("Image", show_image)
            print "Update"
            update = False

        key = cv2.waitKey(0)
        if key > ord('0') and key <= ord('9'):
            return images[key-48-1]
        elif key == ord('q'):
            run = False
