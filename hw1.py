import urllib.request as rq
import numpy as np
import matplotlib.pyplot as plt

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

def prediction_error(X, beta, Y):
    s = np.linalg.norm(np.dot(X,beta)-Y, ord=1)
    return s/(X.shape[0])

def append_ones(X):
    ones = np.full(X.shape[0], 1).reshape(-1,1)
    return np.append(ones, X, axis=1)

def square_loss(X, beta, Y):
    #loss for linear regresion
    error = (np.dot(X, beta)-Y)
    return np.sum(error*error)

def learn_optimal(X, Y):
    #optimal regression
    XT = np.transpose(X)
    XTXinv = np.linalg.inv(np.dot(XT, X))
    return np.dot(np.dot(XTXinv, XT), Y)

def learn_grad_desc(X, Y, iterations, step_size):
    #gradient descent
    loss_history = []
    beta = beta_init(X.shape[1])
    for i in range(iterations):
        ls = square_loss(X, beta, Y)
        grad = sqloss_gradient(X, Y, beta)
        beta = beta - step_size * grad
        loss_history.append(ls)

    return (beta, loss_history)

def beta_init(n):
    return np.full(n, 0)

def sqloss_gradient(X, Y, beta):
    return np.dot(np.transpose(X), np.dot(X, beta) - Y)

def learn_coord_desc(X, Y, iterations):
    #coordinate descent
    loss_history = []
    beta = beta_init(X.shape[1])
    for i in range(iterations):
        ls = square_loss(X, beta, Y)
        k = i%(beta.shape[0])
        bo = coordinate_optimal(X, Y, beta, k)
        beta[k] = bo
        loss_history.append(ls)
    
    return (beta, loss_history)

def coordinate_optimal(X, Y, beta, k):
    Ak = np.dot(X,beta) - Y - X[:,k]*beta[k]
    return -(np.dot(Ak, X[:,k])/np.dot(X[:,k],X[:,k]))

def report_values():
    fullX, fullY = getFullData()
    results = [[],[],[],[]]
    REP_NUM = 10
    for i in range(REP_NUM):
        trainX, trainY, testX, testY = splitData(fullX, fullY)
        trainXones = append_ones(trainX)
        testXones = append_ones(testX)

        # problem 1_1 : optimal with bias term
        beta1_1 = learn_optimal(trainXones, trainY)
        results[0].append(prediction_error(testXones, beta1_1, testY))

        # problem 1_2 : optimal without bias term
        beta1_2 = learn_optimal(trainX, trainY)
        results[1].append(prediction_error(testX, beta1_2, testY))

        # problem 2 : gradient descent
        gd_iter = 100
        gd_step = 0.0008
        beta2 = learn_grad_desc(trainXones, trainY, gd_iter, gd_step)[0]
        results[2].append(prediction_error(testXones, beta2, testY))

        # problem 3 : coordinate descent
        cd_iter = 100
        beta3 = learn_coord_desc(trainXones, trainY, cd_iter)[0]
        results[3].append(prediction_error(testXones, beta3, testY))

    print("-- Prediction Errors --")
    print("<analytic, with bias term>")
    print_data(results[0])
    print("<analytic, bias fixed to 0>")
    print_data(results[1])
    print("<gradient descent>")
    print_data(results[2])
    print("<coordinate descent>")
    print_data(results[3])

def print_data(data_list):
    for data in data_list:
        print("%.3f"%(data))
    print("average: %.3f"%(sum(data_list)/len(data_list)))

def report_graph():
    fullX, fullY = getFullData()
    trainX, trainY, testX, testY = splitData(fullX, fullY)
    trainXones = append_ones(trainX)
    testXones = append_ones(testX)

    # optimal loss
    beta_optimal = learn_optimal(trainXones, trainY)
    loss_optimal = square_loss(trainXones, beta_optimal, trainY)

    # problem 2 : gradient descent
    gd_iter = 100
    history2_small = learn_grad_desc(trainXones, trainY, gd_iter, 0.000001)[1]
    history2_proper = learn_grad_desc(trainXones, trainY, gd_iter, 0.0008)[1]
    history2_large = learn_grad_desc(trainXones, trainY, gd_iter, 0.0011)[1]
    base = np.arange(gd_iter)
    plt.figure(1, figsize=(12,10))
    plt.subplot(221)
    plt.plot(base, np.full(gd_iter, loss_optimal), base, history2_small)
    plt.title('small step size')
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.subplot(222)
    plt.plot(base, np.full(gd_iter, loss_optimal), base, history2_proper)
    plt.title('proper step size')
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.subplot(223)
    plt.plot(base, np.full(gd_iter, loss_optimal), base, history2_large)
    plt.title('large step size')
    plt.ylabel('loss')
    plt.xlabel('iterations')

    # problem 3 : coordinate descent
    cd_iter = 100
    history3 = learn_coord_desc(trainXones, trainY, cd_iter)[1]
    base = np.arange(cd_iter)
    plt.subplot(224)
    plt.plot(base, np.full(gd_iter, loss_optimal), base, history3)
    plt.title('coordinate descent')
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.show()

def main():
    report_values()
    report_graph()

main()