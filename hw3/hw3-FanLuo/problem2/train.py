import numpy as np
import string, random
from collections import Counter
import getopt, sys
import pickle
import datetime

def train():
    with open('train.tagged', 'r') as f:
        lines = [line for line in f]

        word_tag_counts = Counter()                     #initialize
        word_counts = Counter()
        tag_counts = Counter()
        bi_tag_counts = Counter()
        start_tag_counts = Counter() 

        words = []
        tags = []
        bi_tags = []
        word_tags = []
        start_tags = []

        first_word = 1;    #is the first word of a sentence

        for line in lines:                
            if(line.isspace() == False):   #not empty line
                if(first_word == 0):
                    pre_tag = tag
                
                line=line.strip('\n') 
                word = line.split()[0]
                tag = line.split()[1]
                words.append(word)
                tags.append(tag)  
                word_tags.append(line)

                if(first_word == 0):        #not the first word of a sentence
                    bi_tag = pre_tag + '\t' + tag
                    bi_tags.append(bi_tag)
                else:
                    start_tags.append(tag)  #tag of the first word in a sentence
                    first_word = 0
                    
            elif(first_word == 0):          #empty line
                first_word = 1
      

        word_counts.update(words)
        tag_counts.update(tags)
        word_tag_counts.update(word_tags)
        bi_tag_counts.update(bi_tags)
        start_tag_counts.update(start_tags) 


        # save model
        model = []
        model.append(word_counts)
        model.append(tag_counts)
        model.append(word_tag_counts)
        model.append(bi_tag_counts)
        model.append(start_tag_counts) 

        with open('model.txt', "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    train()
    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print("minutes:"+str(minutes)+" seconds:"+str(seconds))