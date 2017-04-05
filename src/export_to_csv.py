# export_to_csv.py
# This is programe use to export data to csv
from helper import list_item, resize, huMonents, logarit, argmax, NAME_LIST
import cv2
import numpy as np
import os
import pandas as pd


def output_layer(n, m):
    arr = np.zeros((m), dtype=np.float32)
    arr[n] = 1.0
    return arr


def inc(max):
    for i in range(max):
        yield i


def generate_label(NAME_LIST):
    label = {}
    m = len(NAME_LIST)
    iterator = inc(m)
    for item in NAME_LIST:
        label[item] = output_layer(next(iterator), m)
    return label


def export_data_to_train(shuffer=False):
    check_data = []

    m = len(NAME_LIST)
    # name = "data_{:02}_suffer".format(m)
    name = "data_{:02}_no_suffer".format(m)

    label = generate_label(NAME_LIST)

    folder_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)

    file_input = os.path.join(folder_data, "data_in.csv")
    file_output = os.path.join(folder_data, "data_out.csv")
    file_check = os.path.join(folder_data, "data_check.csv")

    if not os.path.exists(folder_data):
        os.makedirs(folder_data)

    for index, item_path in enumerate(list_item()):
        name = os.path.split(os.path.dirname(item_path))[1]
        origin = cv2.imread(item_path)
        image = resize(origin)
        hu = huMonents(image).flatten().tolist()
        log = logarit(hu)

        for i in label[name].tolist():
            log.append(i)
        log.append(item_path)
        log.append(argmax([label[name]])[0])
        check_data.append(log)

    if shuffer:
        np.random.shuffle(check_data)

    check_data = np.array(check_data)

    data_in = check_data[:, 0:7]
    data_out = check_data[:, 7:7+m]

    df = pd.DataFrame(data_in)
    df.to_csv(file_input, header=False, index=False)

    df = pd.DataFrame(data_out)
    df.to_csv(file_output, header=False, index=False)

    df = pd.DataFrame(check_data)
    df.to_csv(file_check, header=False, index=False)

    # input = pd.read_csv(file_input, usecols=range(1,8))
    # print(input)

export_data_to_train()
