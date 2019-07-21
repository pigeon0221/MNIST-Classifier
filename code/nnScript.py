import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1 / (1 + np.exp(-1 * z))

sfeatures=None

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    # Your code here.

    trainingSetSize = 50000
    validationSetSize = 10000
    testDataSize = 10000
    columns = 784
    train_data = np.zeros([trainingSetSize, columns])
    train_label = np.zeros([trainingSetSize, ])
    validation_data = np.zeros([validationSetSize, columns])
    validation_label = np.zeros([validationSetSize, ])
    test_data = np.zeros([testDataSize, columns])
    test_label = np.zeros([testDataSize, ])

    trainIndex = 0
    validIndex = 0
    testIndex = 0

    for i in range(0, 10):
        labelTrain = 'train' + str(i)
        labelTest = 'test' + str(i)
        label = mat.get(labelTrain)
        labelT = mat.get(labelTest)
        indexRange = np.random.permutation(range(label.shape[0]))
        indexRangeT = (range(labelT.shape[0]))
        trainRange = indexRange[:label.shape[0] - 1000]
        validRange = indexRange[label.shape[0] - 1000:]
        train_data[trainIndex:trainIndex + len(trainRange)] = label[trainRange, :]
        train_label[trainIndex:trainIndex + len(trainRange)] = i
        validation_data[validIndex:validIndex + len(validRange)] = label[validRange, :]
        validation_label[validIndex:validIndex + len(validRange)] = i
        test_data[testIndex:testIndex + len(indexRangeT)] = labelT[indexRangeT]
        test_label[testIndex:testIndex + len(indexRangeT)] = i
        trainIndex += len(trainRange)
        validIndex += len(validRange)
        testIndex += len(indexRangeT)

    # Deletes 0 columns based on train data
    zeros = np.where(~train_data.any(axis=0))[0]
    global sfeatures
    sfeatures = zeros
    train_data = np.delete(train_data, zeros, axis=1)
    validation_data = np.delete(validation_data, zeros, axis=1)
    test_data = np.delete(test_data, zeros, axis=1)
    train_data = np.double(train_data) / 255.0
    validation_data = np.double(validation_data) / 255.0
    test_data = np.double(test_data) / 255.0
    # selected features = all columns- zeros columns index

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def errorFunction(N, yi, oi):
    # Formula 3.2.3
    LNoi = np.log(oi)
    LN1 = np.log(1 - oi)
    parameter1 = np.multiply(yi, LNoi.T)
    parameter2 = np.multiply(1 - yi, np.transpose(LN1))
    error = parameter1 + parameter2
    error = np.sum(error)
    error = -1 / N * error
    return (error)


def objectiveFunction(Jw1w2, lambdaval, N, w1, w2):
    # Formula 3.2.4
    parameter1 = np.sum(np.multiply(w2, w2))
    parameter2 = np.sum(np.multiply(w1, w1))
    objVal = Jw1w2 + (lambdaval / (2 * N) * (parameter2 + parameter1))
    return (objVal)


def oneOfK(trainDataSize, n_class, training_label):
    yi = np.zeros((trainDataSize, n_class))
    for i in range(0, trainDataSize):
        index = int(training_label[i])
        yi[i][index] = 1
    return (yi)


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


    """
    Folder code must contains the following updated files: nnScript.py and params.pickle 1 . File params.pickle
    contains the learned parameters of Neural Network. Concretely, file params.pickle must contain the
    following variables: list of selected features obtained after feature selection step (selected features), op-
    timal n hidden (number of units in hidden layer), w1 (matrix of weight W (1) as mentioned in section
    3.2.1), w2 (matrix of weight W (2) as mentioned in section 3.2.1), optimal λ (regularization coeffient λ
    as mentioned in section 3.2.4). 2
    """

    trainDataSize = training_data.shape[0]

    # Send from input to hidden nodes
    bias = np.ones([trainDataSize, 1])
    inputWithBias = np.append(training_data, bias, axis=1)
    inputWithBias = np.transpose(inputWithBias)
    hiddenNodes = np.dot(w1, inputWithBias)
    hiddenNodesValue = sigmoid(hiddenNodes)

    # Send from hidden nodes to output
    hiddenWithBias = np.append(hiddenNodesValue, np.transpose(bias), axis=0)
    outputNodesValue = np.dot(w2, hiddenWithBias)
    outputNodesValue = sigmoid(outputNodesValue)

    # 1-of-K coding
    yi = oneOfK(trainDataSize, n_class, training_label)

    # ObjectiveValue & Regularization
    Jw1w2 = errorFunction(trainDataSize, yi, outputNodesValue)
    obj_val = objectiveFunction(Jw1w2, lambdaval, trainDataSize, w1, w2)

    # Gradient
    outputDelta = outputNodesValue - np.transpose(yi)
    outputWithWeights = np.dot(w2.T, outputDelta)
    hiddenWithBiasMinus = np.multiply(hiddenWithBias, (1 - hiddenWithBias))
    hiddenDelta = np.multiply(hiddenWithBiasMinus, outputWithWeights)
    grad_w1 = np.dot(hiddenDelta, inputWithBias.T)
    grad_w1 = grad_w1[0:n_hidden, :]
    grad_w1 = (grad_w1 + lambdaval * w1) / trainDataSize
    grad_w2 = np.dot(outputDelta, hiddenWithBias.T)
    grad_w2 = (grad_w2 + lambdaval * w2) / trainDataSize

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image

    % Output: 
    % label: a column vector of predicted labels"""

    labels = []
    for input_layer in data:
        input_layer_with_bias = np.concatenate((input_layer, [0]))
        hidden_layer = sigmoid(w1 @ input_layer_with_bias)
        hidden_layer_with_bias = np.concatenate((hidden_layer, [0]))
        output_layer = sigmoid(w2 @ hidden_layer_with_bias)
        label = np.argmax(output_layer)
        labels.append(label)

    return np.array(labels)




"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 15

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


all = np.arange(784)
sf = [i for i in all if i not in sfeatures]
DATA = {"n_hidden": n_hidden, "lambda": lambdaval, "selected_features": sf, "w1": w1, "w2": w2}

pickle.dump(DATA, open("params.pickle", "wb"))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
