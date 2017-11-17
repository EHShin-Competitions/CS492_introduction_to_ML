import numpy as np
import urllib.request as rq
import pickle
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix

def main():
    #trainX, trainY, trainYzo, testX, testY, testYzo = get_data()
    #train_partition = k_partition(trainX, trainY, 5)

    # save data
    #DATA = (trainX, trainY, trainYzo, testX, testY, testYzo, train_partition)
    #f = open('dataset.ml', 'wb')
    #pickle.dump(DATA, f)
    #f.close()

    # load data
    f = open('dataset.ml', 'rb')
    DATA = pickle.load(f)
    trainX, trainY, trainYzo, testX, testY, testYzo, train_partition = DATA

    #control
    P1 = False


    #Problem 1
    if(P1):
        logistic_histories = []
        logistic_pred_errors = []
        learning_rates = [0.00003, 0.0003, 0.003]
        for lr in learning_rates:
            beta, history = train_logistic(trainX, trainYzo, lr, gd_iter)
            pred_error = test_logistic(testX, testYzo, beta)
            logistic_histories.append(history)
            logistic_pred_errors.append(pred_error)
        print(logistic_pred_errors)
        graph_base = np.arange(gd_iter)
        plt.figure(1, figsize=(12,10))
        plt.subplot(221)
        plt.plot(graph_base, logistic_histories[0])
        plt.title('small step size')
        plt.ylabel('loss')
        plt.xlabel('iterations')
        plt.subplot(222)
        plt.plot(graph_base, logistic_histories[1])
        plt.title('proper step size')
        plt.ylabel('loss')
        plt.xlabel('iterations')
        plt.subplot(223)
        plt.plot(graph_base, logistic_histories[2])
        plt.title('large step size')
        plt.ylabel('loss')
        plt.xlabel('iterations')
        plt.show()


    #Problem 2
    '''
    linSVM_cand_C = [100, 1000, 10000]
    # 1:5 is good
    for C in linSVM_cand_C:
        ve, te = cross_validate_SVM(train_partition, linear_matrix, [C])
        print("%f:%f:%f"%(C, ve, te))
    '''

   
    #gaussSVM_cand_C = [0.25, 0.5, 1, 2, 4]
    #gaussSVM_cand_sigma = [1,3,5]
    gaussSVM_cand_C = [1000,10000]
    gaussSVM_cand_sigma = [10]
    for C in gaussSVM_cand_C:
        for sigma in gaussSVM_cand_sigma:
            ve, te = cross_validate_SVM(train_partition, gaussian_matrix, [C, sigma])
            print("%f:%f:%f:%f"%(C, sigma, ve, te))

    #avgerr = cross_validate_SVM(train_partition, linear_matrix, [0.01, 2])
    #print(avgerr)

    #Problem 3
    #for different C
    #   for dfiferent sigma
    #       avgerrors3.append cross validation SVM gaussian kernel
    #error3 = test SVM gaussian kernel for best C and sigma

    # report errors
    # draw graphs
    pass

def cross_validate_SVM(train_partition, K_func, hyperparams):
    pred_err_validate = []
    pred_err_training = []
    for i in range(len(train_partition)-3):
        validateX, validateY = train_partition[i]
        trainX, trainY = np.array([]).reshape(0,validateX.shape[1]), np.array([])
        for j in range(len(train_partition)):
            if(i!=j):
                partX, partY = train_partition[j]
                trainX = np.append(trainX, partX, axis=0)
                trainY = np.append(trainY, partY)
        kSVM = train_SVM(trainX, trainY, K_func, hyperparams)
        predY = kSVM.prediction(validateX)
        num_correct = np.sum(predY == validateY)
        pred_err = 1 - (num_correct/validateY.shape[0])
        pred_err_validate.append(pred_err)
        predY = kSVM.prediction(trainX)
        num_correct = np.sum(predY == trainY)
        pred_err = 1 - (num_correct/trainY.shape[0])
        pred_err_training.append(pred_err)

    avg_validate = sum(pred_err_validate)/len(pred_err_validate)
    avg_training = sum(pred_err_training)/len(pred_err_training)
    return avg_validate, avg_training

# kernel matrix functions:
# (i-th row, j-th column) = K(i-th row of X1, j-th row of X2)

def linear_matrix(X1, X2, hyperparams):
    return np.dot(X1, X2.transpose())

def gaussian_matrix(X1, X2, hyperparams):
    X1e = X1.reshape(X1.shape[0], 1, -1)
    X2e = X2.reshape(X2.shape[0], 1, -1)
    D = X1e - X2e.transpose([1,0,2]) #utilize broadcasting
    E = np.exp( -np.sum(D*D, axis=2) / (2*(hyperparams[1]**2)))
    return E

def train_SVM(trainX, trainY, K_func, hyperparams):
    C = hyperparams[0]
    num_data = trainX.shape[0]
    K = K_func(trainX, trainX, hyperparams).astype(np.double)
    P = matrix(trainY*K*(trainY.reshape(-1,1)))
    q = -matrix(np.full(trainX.shape[0], 1).astype(np.double))
    h = matrix((np.append(np.full(trainX.shape[0], 0),np.full(trainX.shape[0], C))).astype(np.double))
    I = np.eye(trainX.shape[0])
    G = matrix(np.append(-I, I, axis=0))
    A = matrix(trainY.reshape(1,-1).astype(np.double))
    B = matrix(np.array([0]).astype(np.double))
    solvers.options['show_progress'] = False
    alpha = solvers.qp(P, q, G, h, A, B)['x']
    #postpone calculating w, b even for linear kernel for consistency
    return trained_SVM(np.array(alpha), trainX, trainY, K_func, hyperparams)

class trained_SVM():
    def __init__(self, alpha, trainX, trainY, K_func, hyperparams):
        self.alpha = alpha
        self.trainX = trainX
        self.trainY = trainY
        self.K_func = K_func
        self.hyperparams = hyperparams
        self.calc_b()

    def prediction(self, testX):
        #predict labels of testX
        K = self.K_func(self.trainX, testX, self.hyperparams)
        weights = self.alpha.reshape(-1) * self.trainY
        WdotPhis = np.dot(weights, K)
        res = np.sign(WdotPhis + self.b)
        return res

    def calc_b(self):
        C = self.hyperparams[0]
        eps = C/10.0
        lb = self.alpha > eps
        ub = self.alpha < C-eps
        support_map = np.logical_and(lb, ub)
        pivot_idx = np.where(support_map)[0][0]
        pivotX = (self.trainX[pivot_idx]).reshape(1,-1)
        K = self.K_func(self.trainX, pivotX, self.hyperparams)
        weights = self.alpha.reshape(-1) * self.trainY
        b = self.trainY[pivot_idx] - np.dot(weights, K)[0]
        self.b = b
        pass

# logistic regression by gradient descent
def train_logistic(trainX, trainYzo, step_size, iterations):
    history = []
    beta = np.full(trainX.shape[1], 0);
    for i in range(iterations):
        history.append(loss_logistic(trainX, trainYzo, beta))
        beta = beta - step_size*grad_logistic(trainX, trainYzo, beta)
    return beta, history

def loss_logistic(trainX, trainYzo, beta):
    P = 1/(1+np.exp(-np.dot(trainX, beta)))
    loss = np.sum(-trainYzo*np.log(P) - (1-trainYzo)*np.log(1-P))
    return loss

# gradient of logistic loss
def grad_logistic(trainX, trainYzo, beta):
    P = 1/(1+np.exp(-np.dot(trainX, beta)))
    grad = -np.dot(trainYzo-P, trainX)
    return grad

def test_logistic(testX, testYzo, beta):
    decision = np.dot(testX, beta) > 0
    num_correct = np.sum(decision == testYzo)
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
    trainYzo = []
    for line in Ltrain:
        y, yzo, x = parse_data_line(line)
        trainX.append(x)
        trainY.append(y)
        trainYzo.append(yzo)

    testX = []
    testY = []
    testYzo = []
    for line in Ltest:
        y, yzo, x = parse_data_line(line)
        testX.append(x)
        testY.append(y)
        testYzo.append(yzo)

    # shuffling train data
    indicies = np.random.permutation(train_size)
    shuffTrainX = np.array(trainX)[indicies]
    shuffTrainY = np.array(trainY)[indicies]
    shuffTrainYzo = np.array(trainYzo)[indicies]

    return shuffTrainX, shuffTrainY, shuffTrainYzo, np.array(testX), np.array(testY), np.array(testYzo)


def parse_data_line(line):
    word_list = line.rstrip().split()
    label = int(word_list[0])
    if(label == -1):
        labelzo = 0
    else:
        labelzo = 1
    encode = []
    for i in range(123):
        encode.append(0)
    word_list = word_list[1:]
    for word in word_list:
        pair = [int(d) for d in word.split(":")]
        if(pair[1] != 0):
            encode[pair[0]] = pair[1]
    return (label, labelzo, encode)

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