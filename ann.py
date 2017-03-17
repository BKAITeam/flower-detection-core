# ann.py
# This is main process
from sklearn.neural_network import MLPClassifier
from helper import list_item, resize, save_img, rename, showImage, huMonents, logarit
from setting import INPUT_PATH
import cv2
import math
import os
import numpy as np

label = {
    "Hoa Canh Buom": [0,0,0,0],
    # "Hoa Hong Mon": [0,0,0,1],
    # "Hoa Mao Ga": [0,0,1,0],
    # "Hoa Sen": [0,0,1,1],
    "Hoa Thien Dieu": [0,1,0,0]
}
label = {
    "Hoa Canh Buom": 0,
    "Hoa Hong Mon": 1,
    "Hoa Mao Ga": 2,
    "Hoa Sen": 3,
    "Hoa Thien Dieu": 4
}
num_of_flower = {
    "Hoa Canh Buom": 0,
    "Hoa Hong Mon": 0,
    "Hoa Mao Ga": 0,
    "Hoa Sen": 0,
    "Hoa Thien Dieu": 0
}
train_data = []
input = []
output = []
list_item(INPUT_PATH)
for index,item_path in enumerate(list_item(INPUT_PATH)):
    name = os.path.split(os.path.dirname(item_path))[1]
    image = cv2.imread(item_path)
    hu = huMonents(image).flatten().tolist()
    log = logarit(hu)
    # train_data.append((log, label[name]))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = image.flatten().tolist()
    # print(image)
    # image = np.array(image, dtype=np.float64).tolist()
    num_of_flower[name] += 1
    input.append(log)
    output.append(label[name])

# print(input)
# 100 ~ 44%
# 90 ~ 39%
# 80 ~ 44%
# 70 ~ 42%
# 60 ~ 39%
# 50 ~ 36%
mlp = MLPClassifier(
    activation="logistic",
    solver='sgd',
    tol=1e-4,
    hidden_layer_sizes=(80,),
    alpha=1e-4,
    random_state=1,
    learning_rate_init=0.001,
    max_iter=10000)

# mlp.fit(input, output)
# score = mlp.score(input, output)
# print("Train score", score)
#
for item in num_of_flower:
    num = num_of_flower[item]
    print(">>>>>>>>>>>>>",item)
    for x in xrange((num)/20+1):
        s = x*20
        f = x*20+20-1 if x*20+20-1 < num else num
        In_Train = input[:s] + input[f:]
        In_Test = input[s:f]
        Out_Train = output[:s] + output[f:]
        Out_Test = output[s:f]
        mlp.fit(In_Train, Out_Train)
        predict = mlp.predict(In_Test)
        score = mlp.score(input, output)
        print(score)
