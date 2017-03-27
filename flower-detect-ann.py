import tensorflow as tf
import numpy as np
from helper import argmax
import pandas as pd
import os
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def input_data(name_data):
    folder_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name_data)

    data_input = pd.read_csv(folder_data + "/data_in.csv", header=None).as_matrix()
    data_output = pd.read_csv(folder_data + "/data_out.csv", header=None).as_matrix()
    return data_input, data_output


num_of_test = 40

# data_input = mnist.train.images
# data_output = mnist.train.labels

# test_x = mnist.test.images
# test_y = mnist.test.labels


def ANN_Train(data_input, data_output, train, save, n_input=7, n_output=13, n_hidden_1 = 40, batch_size=100, learning_rate = 0.001, training_epochs = 200, checkpoint_file="./backup/ann.chk"):

    train_x = data_input[:-num_of_test]
    train_y = data_output[:-num_of_test]


    test_x  = data_input[-num_of_test:]
    test_y  = data_output[-num_of_test:]

    print(train_x.shape)
    print(train_y.shape)

    print(test_x.shape)
    print(test_y.shape)

    display_step = 1

    num_of_train_set = train_x.shape[0]

    print("NUM >>> ", num_of_train_set)

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])


    def newtron_network(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

        return out_layer

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_output]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    net = newtron_network(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        if train:
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_pacth = int(num_of_train_set/batch_size)
                for i in range(total_pacth):
                    bacth_x = train_x[i*batch_size:i*batch_size+batch_size]
                    bacth_y = train_y[i*batch_size:i*batch_size+batch_size]
                    _, c = sess.run([optimizer, cost], feed_dict = {x: bacth_x,
                                                                y: bacth_y})
                    avg_cost += c / total_pacth
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Optimization Finished!")       
            if save:
                saver.save(sess, checkpoint_file)         
        else:
            saver.restore(sess, checkpoint_file)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy with train data:", accuracy.eval({x: train_x, y:train_y}))
        print("Accuracy with test data:", accuracy.eval({x: test_x, y:test_y}))

        prediction=tf.argmax(net,1)
        # n = 482
        print("predictions", argmax(test_y))
        print("predictions", prediction.eval(feed_dict={x: test_x}, session=sess))        


def ANN_Test(data_input, data_output, train, save, n_input=7, n_output=13, n_hidden_1 = 40, batch_size=100, learning_rate = 0.001, training_epochs = 200, checkpoint_file="./backup/ann.chk"):
    train_x = data_input[:-num_of_test]
    train_y = data_output[:-num_of_test]

    test_x  = data_input[-num_of_test:]
    test_y  = data_output[-num_of_test:]

    display_step = 1

    num_of_train_set = train_x.shape[0]

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])

    def newtron_network(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

        return out_layer

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_output]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    net = newtron_network(x, weights, biases)
    
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, checkpoint_file)

        # total_pacth = int(num_of_train_set/batch_size)

        # for i in range(total_pacth):
        #     bacth_x = train_x[i*batch_size:i*batch_size+batch_size]
        #     bacth_y = train_y[i*batch_size:i*batch_size+batch_size]
        #     print(argmax(bacth_y)[0])
        #     correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #     print("Accuracy: %s",i , accuracy.eval({x: bacth_x, y:bacth_y}))

        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy train: ", accuracy.eval({x: train_x, y:train_y}))
        print("Accuracy test: ", accuracy.eval({x: test_x, y:test_y}))

        prediction=tf.argmax(net,1)
        print("predictions", argmax(test_y))
        print("predictions", prediction.eval(feed_dict={x: test_x}, session=sess))


name_backup = "backup_05_suffer_test"

backup_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup", name_backup)
backup_file = os.path.join(backup_folder, "checkpoint.chk")

if not os.path.exists(backup_folder):
    os.makedirs(backup_folder)

name_data = "data_05_suffer"

option = {
 "n_input": 7, 
 "n_output": 5,
 "n_hidden_1": 100,
 "batch_size": 20,
 "learning_rate": 0.001,
 "training_epochs": 500,
 "checkpoint_file":backup_file   
}

data_input, data_output = input_data(name_data)

ANN_Train(data_input, data_output, True, True, **option)

# ANN_Test(data_input, data_output, False, True, **option)