import tensorflow as tf
import numpy as np
from helper import argmax
import pandas as pd



data_input = pd.read_csv("./data_13_suffer/data_in.csv", usecols=range(1,8)).as_matrix()
data_output = pd.read_csv("./data_13_suffer/data_out.csv", usecols=range(1,14)).as_matrix()

# print(data_input)
# print(data_output)

def ANN_Train(data_input, data_output, train, save, checkpoint_file="./backup/ann.chk"):
    train_x = data_input[:-20]
    train_y = data_output[:-20]


    test_x  = data_input[-20:]
    test_y  = data_output[-20:]

    print(test_x[1].shape)
    print(test_y.shape)

    learning_rate = 0.001
    training_epochs = 300000
    batch_size = 100
    display_step = 100

    num_of_train_set = train_x.shape[0]

    print("NUM >>> ", num_of_train_set)

    n_hidden_1 = 176
    n_input = 7
    n_output = 13


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

        for i in range(total_pacth):
            bacth_x = train_x[i*batch_size:i*batch_size+batch_size]
            bacth_y = train_y[i*batch_size:i*batch_size+batch_size]

        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
        print(correct_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: train_x, y:train_y}))

        prediction=tf.argmax(net,1)
        print("predictions", argmax(test_y))
        print("predictions", prediction.eval(feed_dict={x: test_x}, session=sess))


def ANN_Test(data_input, data_output, train, save, checkpoint_file="./backup/ann.chk"):
    train_x = data_input
    train_y = data_output

    test_x  = data_input[-20:]
    test_y  = data_output[-20:]

    batch_size = 200
    display_step = 100

    num_of_train_set = train_x.shape[0]

    n_hidden_1 = 176
    n_input = 7
    n_output = 13


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

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(init)
        saver.restore(sess, checkpoint_file)

        total_pacth = int(num_of_train_set/batch_size)

        # for i in range(total_pacth):
        #     bacth_x = train_x[i*batch_size:i*batch_size+batch_size]
        #     bacth_y = train_y[i*batch_size:i*batch_size+batch_size]
        #     print(argmax(bacth_y)[0])
        #     correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #     print("Accuracy: %s",i , accuracy.eval({x: bacth_x, y:bacth_y}))

        prediction=tf.argmax(net,1)
        n = 482
        print("predictions", argmax(train_y)[n])
        print("predictions", prediction.eval(feed_dict={x: [train_x[n]]}, session=sess))

ANN_Train(data_input, data_output, True, True)

# ANN_Test(data_input, data_output, False, True, checkpoint_file="./backup_100000/ann.chk")