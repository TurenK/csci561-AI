import numpy as np
import pandas as pd
from scipy.special import expit
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
import time


class NeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):
        self.learning_rate = learning_rate
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        # Xavier initialization
        self.w1 = np.random.normal(0.0, pow(self.hidden_layer, -0.5), (self.hidden_layer, self.input_layer))
        self.b1 = np.zeros((hidden_layer, 1))
        self.w2 = np.random.normal(0.0, pow(self.output_layer, -0.5), (self.output_layer, self.hidden_layer))
        self.b2 = np.zeros((output_layer, 1))
        pass

    def sigmoid(self, x):
        # return 1 / (1 + np.exp(-x))

        # handle exp overflow
        return expit(x)

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def get_one_hot_label(self, label, size=10):
        one_hot_label = [0.01 for i in range(size)]
        one_hot_label[label] = 0.99
        return np.array(one_hot_label).reshape((size, 1))

    def activation(self, x):
        return self.sigmoid(x)

    # def softmax(self, x):
    #     # temp = x - np.max(x)
    #     # return np.exp(temp) / np.sum(np.exp(temp), keepdims=True)
    #     return softmax(x)
    #
    # def cross_entropy_loss(self, outputs, one_hot_label):
    #     temp = np.sum(-one_hot_label * np.log(outputs))
    #     return temp

    def output_loss(self, outputs, one_hot_labels):
        return one_hot_labels - outputs

    # pre-modify input, same as softmax after do sigmoid
    def modify_input_img(self, input_img):
        return input_img / 255.0 * 0.99 + 0.01

    def train(self, train_img, train_label, pass_weight=0.01):
        # forward updating
        one_hot_label = self.get_one_hot_label(int(train_label))
        input_imgs = self.modify_input_img(np.array(train_img).reshape((len(train_img), 1)))

        hidden_inputs = np.dot(self.w1, input_imgs) + self.b1
        hidden_outputs = self.activation(hidden_inputs)

        output_inputs = np.dot(self.w2, hidden_outputs) + self.b2
        output_outputs = self.activation(output_inputs)

        # calculate loss
        output_loss = self.output_loss(output_outputs, one_hot_label)
        if np.sum(output_loss) <= pass_weight:
            pass

        # backpropagation
        # w2_now = deepcopy(self.w2)
        # w3_now = deepcopy(self.w3)
        # entropy_softmax_derivative = output_outputs - one_hot_label
        # sigmoid_derivative_hidden_2_input = self.sigmoid_derivative(hidden_2_inputs)
        # sigmoid_derivative_hidden_1_input = self.sigmoid_derivative(hidden_1_inputs)
        w2_derivative = np.dot(output_loss*output_outputs*(1-output_outputs), hidden_outputs.T)
        b2_derivative = output_loss*output_outputs*(1-output_outputs)
        w1_derivative = np.dot(np.dot(self.w2.T, output_loss) * hidden_outputs * (1 - hidden_outputs), input_imgs.T)
        b1_derivative = np.dot(self.w2.T, output_loss) * hidden_outputs * (1 - hidden_outputs)

        # w3_derivative = np.dot(entropy_softmax_derivative, hidden_2_outputs.T)
        # b3_derivative = entropy_softmax_derivative
        # w2_derivative = np.dot(np.dot(w3_now.T, entropy_softmax_derivative) * sigmoid_derivative_hidden_2_input, hidden_1_outputs.T)
        # b2_derivative = np.dot(w3_now.T, entropy_softmax_derivative) * sigmoid_derivative_hidden_2_input
        # w1_derivative = np.dot(np.dot(w2_now.T, np.dot(w3_now.T, entropy_softmax_derivative) * sigmoid_derivative_hidden_2_input) * sigmoid_derivative_hidden_1_input,
        #                                        input_imgs.T)
        # b1_derivative = np.dot(w2_now.T, np.dot(w3_now.T,entropy_softmax_derivative) * sigmoid_derivative_hidden_2_input) * sigmoid_derivative_hidden_1_input

        self.w2 += self.learning_rate * w2_derivative
        self.b2 += self.learning_rate * b2_derivative
        self.w1 += self.learning_rate * w1_derivative
        self.b1 += self.learning_rate * b1_derivative

    def test(self, test_image):
        input_imgs = self.modify_input_img(np.array(test_image).reshape((len(test_image), 1)))

        hidden_inputs = np.dot(self.w1, input_imgs) + self.b1
        hidden_outputs = self.activation(hidden_inputs)

        output_inputs = np.dot(self.w2, hidden_outputs) + self.b2
        output_outputs = self.activation(output_inputs)
        output_outputs = output_outputs.reshape(output_outputs.shape[0])
        output_max_index = np.argmax(output_outputs)
        return output_max_index


def load_data(path):
    return np.array(pd.read_csv(path, header=None))


def plot_accuracy(accuracy, validation_accuracy):
    # line chart
    x = [(i+1) for i in range(len(accuracy))]  # x
    y = accuracy  # y
    y_va = validation_accuracy # y_validation
    plt.plot(x, y, 'o-', color='r', label="Train ACCURACY")  # circle-point
    plt.plot(x, y_va, 's-', color='g', label="Validation ACCURACY")  # square-point
    plt.xlabel("epoch")  # name of x
    plt.ylabel("accuracy")  # name of y
    plt.legend(loc="best")  # legend
    plt.show()
    pass


def cross_validate_train(train_img):
    pass


def train_data(train_img, train_label, network, epoch, batchsize=1, pass_weight = 0.01, isValidate = False, validation_img = None, validation_label = None):
    accuracy = []
    validation_accuracy = []
    for epoch in range(epoch):
        epoch_start = time.time()
        for single_label, single_img in zip(train_label, train_img):
            temp_img = deepcopy(single_img)
            temp_label = deepcopy(single_label)
            network.train(temp_img, temp_label, pass_weight=pass_weight)

        if isValidate:
            test_result = []
            for single_label, single_img in zip(train_label, train_img):
                temp_img = deepcopy(single_img)
                test_result.append(network.test(temp_img))
            test_result = np.array(test_result).reshape((len(test_result), 1))
            temp_accuracy = np.mean(np.equal(test_result, train_label))
            accuracy.append(temp_accuracy)
            print('Train Accuracy: ' + str(temp_accuracy))

            validation_result = []
            for single_label, single_img in zip(validation_label, validation_img):
                temp_img = deepcopy(single_img)
                validation_result.append(network.test(temp_img))
            validation_result = np.array(validation_result).reshape((len(validation_result), 1))
            temp_accuracy = np.mean(np.equal(validation_result, validation_label))
            validation_accuracy.append(temp_accuracy)
            print('Validation Accuracy: ' + str(temp_accuracy))

        epoch_end = time.time()
        print('Epoch time: ' + str(epoch_end - epoch_start))
    if isValidate:
        plot_accuracy(accuracy, validation_accuracy)
    return network


def test_data(test_img, network):
    test_result = []
    for single_img in test_img:
        test_result.append([network.test(single_img)])
    return test_result


def write_test_result(test_result, path='test_predictions.csv'):
    result_dataframe = pd.DataFrame(test_result)
    result_dataframe.index.name = None
    result_dataframe.to_csv(path, header=False, index=False)
    pass

def test_network():
    train_path = 'train_image.csv'
    train_label_path = 'train_label.csv'
    train_img = load_data(train_path)
    train_label = load_data(train_label_path)

    temp_train_img = train_img[0:300]
    temp_train_label = train_label[0:300]
    temp_validation_img = train_img[300:500]
    temp_validation_label = train_label[300:500]

    print('read successfully')

    pass_weight = 0.01
    input_layer = 784
    hidden_layer = 300
    output_layer = 10
    learning_rate = 0.1
    epoch = 100

    network = NeuralNetwork(input_layer, hidden_layer, output_layer, learning_rate)

    network = train_data(temp_train_img, temp_train_label, network, epoch, isValidate=True, pass_weight=pass_weight,
                         validation_img=temp_validation_img, validation_label=temp_validation_label)

    return network

if __name__ == '__main__':
    start = time.time()
    train_img = load_data(sys.argv[1])
    train_label = load_data(sys.argv[2])
    test_img = load_data(sys.argv[3])
    #test_label = load_data('test_label.csv')

    # test and validate
    # network = test_network()

    print('read successfully')

    pass_weight = 0.01
    input_layer = 784
    hidden_layer = 300
    output_layer = 10
    learning_rate = 0.1
    epoch = 10

    network = NeuralNetwork(input_layer, hidden_layer, output_layer, learning_rate)

    network = train_data(train_img, train_label, network, epoch)

    test_result = test_data(test_img, network)

    #temp_accuracy = np.mean(np.equal(np.array(test_result), test_label))

    #print('Test Accuracy: ' + str(temp_accuracy))

    write_test_result(test_result)

    end = time.time()
    print('ALL Time: ' + str(end - start))
