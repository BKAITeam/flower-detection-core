from helper import list_item, resize, save_img, rename, showImage, huMonents, logarit
import cv2
import os
from setting import INPUT_PATH
import tensorflow as tf


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
input = []
output = []


for index, item_path in enumerate(list_item(INPUT_PATH)):
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


training_epochs = 25
learning_rate = 0.01
batch_size = 100
display_step = 1

num_in = 7
num_out = 5

x = tf.placeholder("float", [None, num_in])

y = tf.placeholder("float", [None, num_out])

W = tf.Variable(tf.zeros([num_in, num_out]))

evidence = tf.matmul(x, W)
b = tf.Variable(tf.zeros([10]))
evidence = tf.matmul(x, W) + b
activation = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = y*tf.lg(activation)
cost = tf.reduce_mean(-tf.reduce_sum(cross_entropy, reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

avg_set = []

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

epoch_set = []

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(len(input) / batch_size)
