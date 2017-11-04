import urllib.request as rq
import numpy as np

FIELDS = 13
N_TRAIN = 400

def getFullData():
    PATH = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale"
    ro = rq.urlopen(PATH)
    L = ro.read().decode("utf-8").split()
    dataX = []
    dataY = []
    current_field = FIELDS + 1
    for w in L:
        if(current_field == FIELDS + 1):
            current_field = 0
            dataX.append([])
            dataY.append(float(w))
        else:
            v = float((w.split(":"))[1])
            dataX[-1].append(v)
        current_field += 1
    return np.array(dataX), np.array(dataY)

def splitData(X, Y):
    indicies = np.random.permutation(X.shape[0])
    train_i, test_i = indicies[:N_TRAIN], indicies[N_TRAIN:]
    train_X, test_X = X[train_i,:], X[test_i,:]
    train_Y, test_Y = Y[train_i], Y[test_i]
    return (train_X, train_Y, test_X, test_Y)

def main():
    fullX, fullY = getFullData() #506 samples
    trainX, trainY, testX, testY = splitData(fullX, fullY)

    #include bias term
    trainXones = append_ones(trainX)
    testXones = append_ones(testX)
    beta1_1 = learn1(trainXones, trainY)
    ls1_1 = loss1(trainXones, beta1_1, trainY)
    pe1_1 = prediction_error(testXones, beta1_1, testY)

    #bias fixed to 0
    beta1_2 = learn1(trainX, trainY)
    ls1_2 = loss1(trainX, beta1_2, trainY)
    pe1_2 = prediction_error(testX, beta1_2, testY)

    print("<with bias> ls: %.3f, pe: %.3f"%(ls1_1, pe1_1))
    print("<w/o  bias> ls: %.3f, pe: %.3f"%(ls1_2, pe1_2))

def prediction_error(X, beta, Y):
    s = np.linalg.norm(np.dot(X,beta)-Y, ord=1)
    return s/(X.shape[0])

def append_ones(X):
    ones = np.full(X.shape[0], 1).reshape(-1,1)
    return np.append(ones, X, axis=1)

def loss1(X, beta, Y):
    #loss for linear regresion
    error = (np.dot(X, beta)-Y)
    return np.sum(error*error)

def learn1(X, Y):
    XT = np.transpose(X)
    XTXinv = np.linalg.inv(np.dot(XT, X))
    return np.dot(np.dot(XTXinv, XT), Y)

main()