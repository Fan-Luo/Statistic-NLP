#!/usr/bin/python3
#author: Fan Luo
import numpy as np
import string
from collections import Counter
import getopt, sys
import math
import pickle
import datetime

def test():

    #load model
    with open('model.txt', 'rb') as m:     
        model = pickle.load(m)
        words = model[0]
        tags = model[1]
        pr_transition = model[2]
        pr_likihood = model[3]
        pr_start_tag = model[4]
        pr_stop_tag = model[5] 

        len_words = len(words)
        len_tags = len(tags)

        with open('test.tagged', 'r') as f:
            lines = [line for line in f]

            match = 0
            unknown_match = 0
            total_word = 0
            unknown_word = 0

            step = 0                                #the first word of each sentence
            viterbi =[]
            backpointer =[]
            gold_tags = []
            unknown_index = []

            for l,line in enumerate(lines):   
                if(line.isspace() == False):        #not empty line 
                    line=line.strip()
                    test_word = line.split()[0]
                    gold_tag = line.split()[1]
                    gold_tags.append(gold_tag)

                    total_word += 1
                    if(test_word not in words):
                        unknown_word += 1
                        unknown_index.append(step)
                        word_index = len_words
                    else:
                        word_index = words.index(test_word)

                    if(step == 0):                  #initialization step
                        viterbi.append(np.zeros(len_tags))
                        viterbi[0] = pr_start_tag * pr_likihood[word_index]
                        backpointer.append(np.zeros(len_tags))  

                    else:                           #recursion step
                        viterbi.append(np.zeros(len_tags))
                        backpointer.append(np.zeros(len_tags))  
                        for t in range(len_tags):
                            v_rec = viterbi[step-1]*pr_transition[t]
                            max_index = np.argmax(v_rec)
                            backpointer[step][t] = max_index
                            viterbi[step][t] = v_rec[max_index] * pr_likihood[word_index][t]

                    step += 1

                if(((line.isspace() == True) or (l == len(lines)-1)) and (step > 0)):     # termintion step

                    v_ter = viterbi[step-1] * pr_stop_tag 
                    max_index = np.argmax(v_ter)
                    backpointer.append([max_index])  
                 
                    #compare predict v.s. gold
                    t = step - 1
                    s = max_index 
                    while(t >= 0):
                        predict_tag = tags[s]
                        s = int(backpointer[t][s]) 

                        if(predict_tag == gold_tags[t]):
                            match += 1
                            if(t in unknown_index):
                                unknown_match += 1
                        t -= 1

                    # initialize for next sentence
                    viterbi =[]
                    backpointer =[]
                    gold_tags = []
                    step = 0
                    unknown_index = []

            accuracy = match / total_word    
            unknown_accuracy = unknown_match / unknown_word   
            print("Overall accuracy: %.2f%%\n" % (100*accuracy))
            print("Accuracy for unknown words: %.2f%%\n" % (100*unknown_accuracy))


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    test()
    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print("minutes:"+str(minutes)+" seconds:"+str(seconds))
