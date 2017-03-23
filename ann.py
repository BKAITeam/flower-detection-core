# ann.py
# This is main process
from sklearn.neural_network import MLPClassifier
from helper import list_item, resize, save_img, rename, showImage, huMonents, logarit
from setting import INPUT_PATH
import cv2
import math
import os
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix

# label = {
#     "Hoa Canh Buom": [0,0,0,0],
#     # "Hoa Hong Mon": [0,0,0,1],
#     # "Hoa Mao Ga": [0,0,1,0],
#     # "Hoa Sen": [0,0,1,1],
#     "Hoa Thien Dieu": [0,1,0,0]
# }

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


list_item(INPUT_PATH)
for index,item_path in enumerate(list_item(INPUT_PATH)):
    name = os.path.split(os.path.dirname(item_path))[1]
    image = cv2.imread(item_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten().tolist()
    hu = huMonents(image).flatten().tolist()
    log = logarit(hu)
    # train_data.append((log, label[name]))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = image.flatten().tolist()
    # print(image)
    # image = np.array(image, dtype=np.float64).tolist()
    num_of_flower[name] += 1
    log.append(label[name])
    train_data.append(log)
    # output.append(label[name])

# np.random.shuffle(train_data)
train_data = np.array(train_data)
input = train_data[:,:7].tolist()
output = train_data[:,7].astype(int).tolist()


mlp = MLPClassifier(
    activation="logistic",
    solver='sgd',
    tol=1e-4,
    hidden_layer_sizes=(176,),
    alpha=1e-5,
    random_state=1,
    early_stopping=True,
    # verbose=True,
    learning_rate_init=0.01,
    max_iter=10000)

# mlp.fit(input[:-20], output[:-20])
# mlp = mlp.fit(input, output)
# predict = mlp.predict(input[-20:])
# print(np.array(output[-20:]))
# print(predict)
# score = mlp.score(input, output)
# print("Train score", score)


sum = 0
for item in num_of_flower:
    num = num_of_flower[item]
    # start = num + num_of_flower[item]
    print(">>>>>>>>>>>>>>>>>>")

    for x in xrange((num)/20+1):
        s = sum + x*20
        f = sum + x*20+20-1 if x*20+20-1 < num else sum + num
        # print(s, f, label[item])

        In_Train = input[:s] + input[f:]
        In_Test = input[s:f]

        Out_Train = output[:s] + output[f:]
        Out_Test = output[s:f]

        mlp = mlp.fit(In_Train, Out_Train)

        predict = mlp.predict(In_Test)
        print(">>>>>>>>>>>>>>>>>")
        print(np.array(Out_Test))
        print(predict)
        score = mlp.score(In_Test, Out_Test)
        print(score)
        print("<<<<<<<<<<<<<<<<<<")
    sum += num
