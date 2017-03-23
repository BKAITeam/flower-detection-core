# export_to_csv.py
# This is programe use to export data to csv
from helper import list_item, resize, save_img, rename, showImage, huMonents, logarit, argmax
from setting import INPUT_PATH
import cv2
import math
import numpy as np
import os
import pandas as pd


def output_layer(n):
    arr = np.zeros((13), dtype=np.float32)
    arr[n] = 1.0
    return arr

label = {
    "Hoa Canh Buom": output_layer(0),
    "Hoa Hong": output_layer(1),
    "Hoa Hong Mon": output_layer(2),
    "Hoa Huong Duong": output_layer(3),
    "Hoa Ly": output_layer(4),
    "Hoa Mao Ga": output_layer(5),
    "Hoa Sen": output_layer(6),
    "Hoa Thien Dieu": output_layer(7),
    "Hoa Thuoc Duoc": output_layer(8),
    "Hoa Trang": output_layer(9),
    "Hoa But": output_layer(10),
    "Hoa Cuc Trang": output_layer(11),
    "Hoa Rum": output_layer(12),
}


file_input = "./data_13_suffer/data_in.csv"
file_output = "./data_13_suffer/data_out.csv"
file_check = "./data_13_suffer/data_check.csv"


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
    data_out = check_data[:,7:20]

    df = pd.DataFrame(data_in)
    df.to_csv(file_input)

    df = pd.DataFrame(data_out)
    df.to_csv(file_output)

    df = pd.DataFrame(check_data)
    df.to_csv(file_check)

    input = pd.read_csv(file_input, usecols=range(1,8))
    print(input)

export_data_to_train()