#!/usr/bin/python3  
#author: Fan Luo

import string
from collections import Counter  

def hasalpha(inputString):
	return any(char.isalpha() for char in inputString)

with open('brown_sample.txt', 'r') as f: 
	content = f.read().lower()						#read content of brown sample.txt
	pairs = content.split()				

	word_tag_counts = Counter()						#initialize
	word_counts = Counter()
	tag_counts = Counter()

	words = []
	tags = []
	word_tags = []

	for pair in pairs:
		split_item = pair.split('/')                #separate word and tag 
		word = split_item[0]
		tag = split_item[1]
		
		if (True == hasalpha(tag) ):    			#a valid tag	
			words.append(word)
			tags.append(tag)  
			word_tags.append(pair)
	
	for word in words:								#count number of each word
		word_counts[word] += 1

	for tag in tags:								#count number of each tag
		tag_counts[tag] += 1

	for word_tag in word_tags:						#count number of each word-tag pair
		word_tag_counts[word_tag] += 1

	Top10_words = word_counts.most_common(10)
	Top10_tags = tag_counts.most_common(10)
	Top10_word_tags = word_tag_counts.most_common(10)

	print ('Top 10 most frequent words |\tFrequency')
	for Top10_word in Top10_words:
		print (Top10_word[0], end='\t')
		print (Top10_word[1])

	print ('\nTop 10 most frequent POS tags |\tFrequency')
	for Top10_tag in Top10_tags:
		print (Top10_tag[0], end='\t')
		print (Top10_tag[1])

	print ('\nTop 10 most frequent word-POS tag pairs |\tFrequency')
	for Top10_word_tag in Top10_word_tags:
		print (Top10_word_tag[0], end='\t')
		print (Top10_word_tag[1])	
	 