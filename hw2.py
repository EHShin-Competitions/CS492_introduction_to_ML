import numpy as np
import urllib.request as rq

def main():
    trainX, trainY, testX, testY = get_data()
    train_partition = k_partition(trainX, trainY, 5)

    #Problem 1
    #for different learning rates
    #   graph1.append train logistic
    #   error1.append test logistic
    logistic_beta, logistic_history = train_logistic(trainX, trainY, 0.0009, 500)
    logistic_error = test_logistic(testX, testY, logistic_beta)

    #Problem 2
    #for different C
    #   avgerrors2.append cross validation SVM linear kernel
    #error2 = test SVM linear kernel for best C

    #Problem 3
    #for different C
    #   for dfiferent sigma
    #       avgerrors3.append cross validation SVM gaussian kernel
    #error3 = test SVM gaussian kernel for best C and sigma

    # report errors
    # draw graphs
    pass

# logistic regression by gradient descent
def train_logistic(trainX, trainY, step_size, iterations):
    history = []
    beta = np.full(trainX.shape[1], 0);
    for i in range(iterations):
        history.append(loss_logistic(trainX, trainY, beta))
        beta = beta - step_size*grad_logistic(trainX, trainY, beta)
    return beta, history

def loss_logistic(trainX, trainY, beta):
    P = 1/(1+np.exp(-np.dot(trainX, beta)))
    loss = np.sum(-trainY*np.log(P) - (1-trainY)*np.log(1-P))
    return loss

# gradient of logistic loss
def grad_logistic(trainX, trainY, beta):
    P = 1/(1+np.exp(-np.dot(trainX, beta)))
    grad = -np.dot(trainY-P, trainX)
    return grad

def test_logistic(testX, testY, beta):
    decision = np.dot(testX, beta) > 0
    num_correct = np.sum(decision == testY)
    pred_error = 1 - (num_correct/testX.shape[0])
    return pred_error

# fetch traning / test data
def get_data():
    URL_TRAIN = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a"
    URL_TEST = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a.t"
    Ltrain = rq.urlopen(URL_TRAIN).read().decode('utf-8').rstrip().split('\n')
    Ltestfull = rq.urlopen(URL_TEST).read().decode('utf-8').rstrip().split('\n')
    Ltest = Ltestfull[:1000]
    train_size = len(Ltrain)
    trainX = []
    trainY = []
    for line in Ltrain:
        y, x = parse_data_line(line)
        trainX.append(x)
        trainY.append(y)

    testX = []
    testY = []
    for line in Ltest:
        y, x = parse_data_line(line)
        testX.append(x)
        testY.append(y)

    # shuffling train data
    indicies = np.random.permutation(train_size)
    shuffTrainX = np.array(trainX)[indicies]
    shuffTrainY = np.array(trainY)[indicies]

    return shuffTrainX, shuffTrainY, np.array(testX), np.array(testY)


def parse_data_line(line):
    word_list = line.rstrip().split()
    label = int(word_list[0])
    ZERO_ONES = True;
    if(ZERO_ONES and label == -1):
        label = 0
    encode = []
    for i in range(123):
        encode.append(0)
    word_list = word_list[1:]
    for word in word_list:
        pair = [int(d) for d in word.split(":")]
        if(pair[1] != 0):
            encode[pair[0]] = pair[1]
    return (label, encode)

def k_partition(trainX, trainY, k):
    # assume pre-shuffled
    b_size = trainX.shape[0]//k
    partition = []
    for i in range(k-1):
        partX = trainX[i*b_size:(i+1)*b_size]
        partY = trainY[i*b_size:(i+1)*b_size]
        partition.append((partX, partY))
    partX = trainX[(k-1)*b_size:]
    partY = trainY[(k-1)*b_size:]
    partition.append((partX, partY))
    return partition


main()