#!/usr/bin/python3
#author: Fan Luo
import numpy as np
import string
from collections import Counter
import getopt, sys
import math
import pickle

def loadmodel(modelfile):
        with open(modelfile, 'rb') as model:     
            content = pickle.load(model)
            word_counts = content[0]
            tag_counts = content[1]
            word_tag_counts = content[2]
            bi_tag_counts = content[3]
            start_tag_counts = content[4] 

            return (word_counts,tag_counts,word_tag_counts,bi_tag_counts,start_tag_counts)

def test():

    #load model
    (C_word,C_tag,C_word_tag,C_bi_tag,C_start_tag) = loadmodel('model.txt')
    tags = list(C_tag)
    words = list(C_word)
    len_tags = len(tags)
    len_words = len(words)

    with open('test.tagged', 'r') as f:
        lines = [line for line in f]

        match = 0
        total = 0
        unknown = 0
        unknown_match =0
        first_word = 1
        for line in lines:     
            if(line.isspace() == False):   #not empty line
                line=line.strip()
                word = line.split()[0]
                tag = line.split()[1]
                if(word not in words):
                    unknown += 1
                if(first_word == 0):
                    pre_tag = predict_tag
                    for t in tags: 
                        bi_tag = pre_tag + '\t' + t
                        pr_transition = (float(C_bi_tag[bi_tag])+1) / (float(C_tag[pre_tag])+len_tags)
                       
                        word_tag = word+'\t'+t
                        pr_likihood = (float(C_word_tag[word_tag])+1)/ (float(C_tag[t])+len_words)
                        
                        if((t == tags[0]) or (pr_transition*pr_likihood > pr)):
                            pr = pr_transition*pr_likihood
                            predict_tag = t

                else:       # fist word of a sentence
                    first_word = 0
                    for t in tags: 
                        pr_transition = (float(C_start_tag[t])+1) / (float(sum(C_start_tag.values()))+len_tags) 
                        word_tag = word+'\t'+t
                        pr_likihood = (float(C_word_tag[word_tag])+1)/ (float(C_tag[t])+len_words)

                        if((t == tags[0]) or (pr_transition*pr_likihood > pr)):
                            pr = pr_transition*pr_likihood
                            predict_tag = t
                
                if(predict_tag == tag):
                    match += 1
                    if(word not in words):
                        unknown_match += 1
                total += 1
            else:
                first_word = 1

        accuracy = match / total    
        unknown_accuracy = unknown_match / unknown   
        print("Overall accuracy: %.2f%%\n" % (100*accuracy))
        print("Accuracy for unknown words: %.2f%%\n" % (100*unknown_accuracy))




if __name__ == '__main__':
    test()
