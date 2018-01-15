#!/usr/bin/python3
#author: Fan Luo
import numpy as np
import string, random
from collections import Counter
import getopt, sys

# Sigmoid function
def sigmoid(z):
    if(z < -20):
        return 0
    else:
        return 1 / (1 + np.exp(-z))


def TrainData_preprocess(datafile, frequency):
    with open(datafile, 'r') as f:
        lines = [line.lower() for line in f]

        Y = []
        word_counts = Counter()
        biword_counts = Counter()
        word_count_list = []
        biword_count_list = []

        for line in lines:                  # each message
            for c in string.punctuation:    # remove all punctuation
                line = line.replace(c, " ")

            if(line.split()[0] == "spam"):
                Y.append(1)
            else:
                Y.append(0)

            message_word_count = Counter()
            message_biword_count = Counter()

            for w in range(len(line.split())):
                if(w > 0):
                    word = line.split()[w]
                    word_counts[word] += 1          # count word's frequency among dataset
                    message_word_count[word] += 1   # count word's frequency among each message
                    if(w < len(line.split()[1:])):
                        biword = line.split()[w] + ' ' + line.split()[w+1]
                        biword_counts[biword] += 1
                        message_biword_count[biword] += 1
            word_count_list.append(message_word_count)
            biword_count_list.append(message_biword_count)

        #features
        words = [word for word in word_counts if (word_counts[word] > frequency) ]
        biwords = [biword for biword in biword_counts if (biword_counts[biword] > frequency) ]

        X = np.zeros((len(lines), len(words)+len(biwords)), dtype=np.int)
        for i in range(len(lines)):
            for j in range(len(words)):
                X[i][j] = word_count_list[i][words[j]]
            for j in range(len(biwords)):
                X[i][j+len(words)] = biword_count_list[i][biwords[j]]

        Y = np.array([Y])
        return(np.hstack((X, Y.T)), words, biwords)


def train(frequency_threshold, learning_rate, epoch_limit, batch_size):

    (train_data, train_words, train_biwords) = TrainData_preprocess('SMSSpamCollection.train', frequency_threshold)
    theta = np.zeros(len(train_words)+len(train_biwords))
    epoch = 0

    #Logistic regression
    for epoch in range(epoch_limit):
        np.random.shuffle(train_data)
        train_X = train_data[:, :-1]
        train_Y = train_data[:, -1]

        gradient = np.zeros(len(train_words)+len(train_biwords))
        for i in range(train_X.shape[0]):
            h = sigmoid(np.dot(train_X[i], theta))
            gradient += np.dot(train_Y[i] - h, train_X[i])
            if( (i+1) % batch_size == 0):
                theta += learning_rate * gradient / batch_size
                gradient = np.zeros(len(train_words)+len(train_biwords))
            elif(i == train_X.shape[0]-1):
                theta += learning_rate * gradient / (train_X.shape[0] % batch_size)


    #save model
    with open('model_bigram.txt', 'w') as m:
        m.write("Selected features | Theta")
        for i in range(len(train_words)):
            m.write("\n'%s'\t%s" % (str(train_words[i]), str(theta[i])))
        for i in range(len(train_biwords)):
            m.write("\n'%s'\t%s" % (str(train_biwords[i]), str(theta[i+len(train_words)])))


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:r:e:s:", ["frequency_threshold=", "learning_rate=", "epoch_limit=", "batch_size="])
    except getopt.GetoptError:
        sys.exit()

    for opt, arg in opts:
        if opt in ("-f", "--frequency_threshold"):
            f = int(arg)
        elif opt in ("-r", "--learning_rate"):
            r = float(arg)
        elif opt in ("-e", "--epoch_limit"):
            e = int(arg)
        elif opt in ("-s", "--batch_size"):
            s = int(arg)
        else:
            assert False, "unhandled option"

    train(f,r,e,s)