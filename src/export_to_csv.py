# export_to_csv.py
# This is programe use to export data to csv
from helper import list_item, resize, save_img, rename, showImage, huMonents, logarit, argmax
from setting import INPUT_PATH
import cv2
import math
import numpy as np
import os
import pandas as pd

m = 15

def output_layer(n, m):
    arr = np.zeros((m), dtype=np.float32)
    arr[n] = 1.0
    return arr

label = {
    "Hoa Canh Buom": output_layer(0, m),
    "Hoa Hong": output_layer(1, m),
    "Hoa Hong Mon": output_layer(2, m),
    "Hoa Huong Duong": output_layer(3, m),
    "Hoa Ly": output_layer(4, m),
    "Hoa Mao Ga": output_layer(5, m),
    "Hoa Sen": output_layer(6, m),
    "Hoa Thien Dieu": output_layer(7, m),
    "Hoa Thuoc Duoc": output_layer(8, m),
    "Hoa Trang": output_layer(9, m),
    "Hoa But": output_layer(10, m),
    "Hoa Cuc Trang": output_layer(11, m),
    "Hoa Rum": output_layer(12, m),
    "Hoa Cam Tu Cau": output_layer(13, m),
    "Hoa Van Tho": output_layer(14, m),
}

name = "data_{:02}_suffer".format(m)
# name = "data_{:02}_no_suffer".format(m)

folder_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)

file_input = os.path.join(folder_data, "data_in.csv")
file_output = os.path.join(folder_data, "data_out.csv")
file_check = os.path.join(folder_data, "data_check.csv")


if not os.path.exists(folder_data):
    os.makedirs(folder_data)

def export_data_to_train():
    check_data = []

    for index,item_path in enumerate(list_item()):
        name = os.path.split(os.path.dirname(item_path))[1]
        origin = cv2.imread(item_path)
        image = resize(origin)
        hu = huMonents(image).flatten().tolist()
        log = logarit(hu)
        log_np = np.array(logarit(hu))

        for i in label[name].tolist():
            log.append(i)
        log.append(item_path)
        log.append(argmax([label[name]])[0])
        check_data.append(log)

    np.random.shuffle(check_data)

    check_data = np.array(check_data)
    
    data_in = check_data[:,0:7]
    data_out = check_data[:,7:7+m]

    # data_in = np.multiply(data_in, 1.0)
    df = pd.DataFrame(data_in)
    df.to_csv(file_input, header=False, index=False)

    df = pd.DataFrame(data_out)
    df.to_csv(file_output, header=False, index=False)

    df = pd.DataFrame(check_data)
    df.to_csv(file_check, header=False, index=False)

    # input = pd.read_csv(file_input, usecols=range(1,8))
    # print(input)

export_data_to_train()