import numpy as np
import urllib.request as rq
import pickle
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

    #Problem 1
    #for different learning rates
    #   graph1.append train logistic
    #   error1.append test logistic
    '''
    logistic_histories = []
    logistic_pred_errors = []
    learning_rates = [0.0001, 0.0003, 0.001, 0.0012]
    for lr in learning_rates:
        beta, history = train_logistic(trainX, trainYzo, lr, 300)
        pred_error = test_logistic(testX, testYzo, beta)
        logistic_histories.append(history)
        logistic_pred_errors.append(pred_error)
    print(logistic_pred_errors)
    '''

    #Problem 2
    #for different C
    #   avgerrors2.append cross validation SVM linear kernel
    #error2 = test SVM linear kernel for best C
    print("start SVM")
    train_SVM(trainX, trainY, linear_P, [1])
    print("end SVM")
    #Problem 3
    #for different C
    #   for dfiferent sigma
    #       avgerrors3.append cross validation SVM gaussian kernel
    #error3 = test SVM gaussian kernel for best C and sigma

    # report errors
    # draw graphs
    pass


def linear_kernel(x1, x2, hyperparams):
    return  np.dot(x1, x2)

def gaussian_kernel(x1, x2, hyperparams):
    diff = x1-x2
    sumSqDiff = np.dot(diff, diff)
    return np.exp(-sumSqDiff/(2*hyperparams[1]))

def linear_P(X, Y, hyperparams):
    T = Y.reshape(-1,1) * X
    return np.dot(T, T.transpose())

def gaussian_P(X, Y, hyperparams):
    Xe = X.reshape(X.shape[0], 1, -1)
    D = Xe - Xe.transpose([1,0,2]) #utilize broadcasting
    E = np.exp( -np.sum(D*D, axis=2) / (2*(hyperparams[1]**2)))
    P = Y*E*(Y.reshape(-1,1))
    return P

def test():
    P = matrix([[14.,5.,8.],[5.,14.,9.],[8.,9.,14.]])
    q = matrix([1.,1.,1.])
    G = matrix([[-1.0,0.0,0.0],
                [0.0,-1.0,0.0],
                [0.0,0.0,-1.0],
                [1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,1.0]]).trans()
    print(G)
    C = 1.
    h = matrix([0.,0.,0.,C,C,C])
    A = matrix([-1.,-1.,-1.]).trans()
    b = matrix([0.])
    a = solvers.qp(P, q, G, h, A, b)

def train_SVM(trainX, trainY, P_func, hyperparams):
    C = hyperparams[0]
    num_data = trainX.shape[0]
    P = P_func(trainX, trainY, hyperparams).astype(np.double)
    P = matrix(P)
    q = -matrix(np.full(trainX.shape[0], 1).astype(np.double))
    h = matrix((np.append(np.full(trainX.shape[0], 0),np.full(trainX.shape[0], C))).astype(np.double))
    I = np.eye(trainX.shape[0])
    G = matrix(np.append(-I, I, axis=0))
    A = matrix(trainY.reshape(1,-1).astype(np.double))
    B = matrix(np.array([0]).astype(np.double))
    alpha = solvers.qp(P, q, G, h, A, B)['x']
    print(alpha)
    #postpone calculating w, b even for linear kernel for consistency
    return trained_SVM(np.array(alpha), trainX, trainY)

class trained_SVM():
    def __init__(self, alpha, trainX, trainY):
        self.alpha = alpha
        self.trainX = trainX
        self.trainY = trainY

    def predict(self, x):
        #predict label of x

        return None

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