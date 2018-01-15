#!/usr/bin/python3  
#author: Fan Luo

import numpy as np

words = np.genfromtxt('vectors_top3000.txt', delimiter=' ', usecols=0, dtype=str)
vectors = np.genfromtxt('vectors_top3000.txt', delimiter=' ')[:,1:]

words_num = words.size
home_index = np.where(words=="home")[0][0]

elementwise_product = np.tile(vectors[home_index],(words_num,1)) * vectors  #multiply each verctor with "home" elementwisely
similarities = elementwise_product.sum(axis = 1)     # summary of first dimensional
similarity_increasingly_Index = np.argsort(similarities)		 
similarity_decreasingly_Index = np.argsort(-similarities)

print ('Top 10 most similar words |\tSimilarity')

i = 0
j = 0
while (i < 10):
	if (similarity_decreasingly_Index[j] != home_index):
		print (words[similarity_decreasingly_Index[j]], end='\t')
		print (similarities[similarity_decreasingly_Index[j]])
		i += 1	
	j += 1

print ('\nTop 10 most dissimilar words |\tSimilarity')

i = 0
j = 0
while (i < 10):
	if (similarity_increasingly_Index[j] != home_index):
		print (words[similarity_increasingly_Index[j]], end='\t')
		print (similarities[similarity_increasingly_Index[j]])
		i += 1	
	j += 1
