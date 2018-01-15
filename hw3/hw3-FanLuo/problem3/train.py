import numpy as np
import string, random
from collections import Counter
import getopt, sys
import pickle
import datetime

def generate_model(C_word,C_tag,C_word_tag,C_bi_tag,C_start_tag,C_stop_tag):

    words = list(C_word)
    tags = list(C_tag)
    len_words = len(words)
    len_tags = len(tags)


    pr_transition = np.zeros((len_tags,len_tags))
    for t, tag in enumerate(tags):
        for pre_t, pre_tag in enumerate(tags):
            bi_tag = pre_tag + '\t' + tag
            pr_transition[t][pre_t] = (float(C_bi_tag[bi_tag])+1) / (float(C_tag[pre_tag])+len_tags)

    pr_likihood = np.zeros((len_words+1, len_tags))
    for t, tag in enumerate(tags):
        for w, word in enumerate(words):
            word_tag = word+'\t'+tag
            pr_likihood[w][t] = (float(C_word_tag[word_tag])+1)/ (float(C_tag[tag])+len_words)
        pr_likihood[len_words][t] = 1.0 / (float(C_tag[tag])+len_words)           #unknown word

    pr_start_tag = np.zeros(len_tags)
    for t, tag in enumerate(tags):
        pr_start_tag[t] = (float(C_start_tag[tag])+1) / (float(sum(C_start_tag.values()))+len_tags)  

    pr_stop_tag = np.zeros(len_tags)
    for t, tag in enumerate(tags):
        pr_stop_tag[t] = (float(C_stop_tag[tag])+1) / (float(sum(C_stop_tag.values()))+len_tags) 

    # save model
    model = []
    model.append(words)
    model.append(tags)
    model.append(pr_transition)
    model.append(pr_likihood)
    model.append(pr_start_tag)
    model.append(pr_stop_tag)

    with open('model.txt', "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def train():
    with open('train.tagged', 'r') as f:
        lines = [line for line in f]

        #initialize
        word_counts = Counter()
        tag_counts = Counter()
        word_tag_counts = Counter()  
        bi_tag_counts = Counter()
        start_tag_counts = Counter()
        stop_tag_counts = Counter()

        words = []
        tags = []
        bi_tags = []
        word_tags = []
        start_tags = []
        stop_tags = []

        first_word = 1;                                             #is the first word of a sentence

        for l,line in enumerate(lines):                
            if(line.isspace() == False):                            #not empty line
                if(first_word == 0):
                    pre_tag = tag
                
                line=line.strip('\n') 
                word = line.split('\t')[0]
                tag = line.split('\t')[1]
                words.append(word)
                tags.append(tag)  
                word_tags.append(line)

                if(first_word == 0):                                #not the first word of a sentence
                    bi_tag = pre_tag + '\t' + tag
                    bi_tags.append(bi_tag)
                else:
                    start_tags.append(tag)                          #tag of the first word in a sentence
                    first_word = 0
                    
            elif(first_word == 0):                                  #empty line   
                stop_tags.append(tag)                               #tag of the last word in a sentence
                first_word = 1
            
            if(l == len(lines)-1):                                  #last line
                stop_tags.append(tag)  

        word_counts.update(words)
        tag_counts.update(tags)
        word_tag_counts.update(word_tags)
        bi_tag_counts.update(bi_tags)
        start_tag_counts.update(start_tags)
        stop_tag_counts.update(stop_tags)
        
        #generate and save model
        generate_model(word_counts,tag_counts,word_tag_counts,bi_tag_counts,start_tag_counts,stop_tag_counts)


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    train()
    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print("minutes:"+str(minutes)+" seconds:"+str(seconds))