import numpy as np
import urllib.request as rq

def main():
    #get data
    #make 5-fold partition

    #Problem 1
    #for different learning rates
    #   graph1.append train logistic
    #   error1.append test logistic

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

# fetch traning / test data
def get_data():
    URL_TRAINING = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a"
    URL_TEST = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a.t"
    Ltraining = rq.urlopen(URL_TRAINING).read().decode('utf-8').rstrip().split('\n')
    Ltestraw = rq.urlopen(URL_TEST).read().decode('utf-8').rstrip().split('\n')
    Ltest = Ltestraw[:1000]


def parse_data_line(line):
    line = '+1 4:1 10:1 16:1 29:1 39:1 40:1 52:1 63:1 67:1 73:1 74:1 77:1 80:1 83:1 '
    word_list = line.rstrip().split()
    label = int(word_list[0])
    ZERO_ONES = False;
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