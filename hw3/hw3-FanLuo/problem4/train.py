import numpy as np
import string, random
from collections import Counter
import getopt, sys
import pickle 
import dynet_config
dynet_config.set(
    mem=16384,
    autobatch=True,      # utilize autobatching
    random_seed=1978     # simply for reproducibility here
)
import dynet as dy   
import datetime


def load_pretrained_embeddings(path_to_file, take):
    embedding_size = 200
    embedding_matrix = None
    lookup = {"<unk>": 0}
    c = 0
    with open(path_to_file, "r") as f:
        delimiter = " "
        for line in f:        
            if (take and c <= take) or not take:
                # split line
                line_split = line.rstrip().split(delimiter)
                # extract word and vector
                word = line_split[0]
                vector = np.array([float(i) for i in line_split[1:]])
                # get dimension of vector
                embedding_size = vector.shape[0]
                # add to lookup
                lookup[word] = c
                # add to embedding matrix
                if np.any(embedding_matrix):
                    embedding_matrix = np.vstack((embedding_matrix, vector))
                else:
                    embedding_matrix = np.zeros((2, embedding_size))
                    embedding_matrix[1] = vector
                c += 1
    return embedding_matrix, lookup

def import_data(train_file):
    with open(train_file, 'r') as f:
        lines = [line for line in f]

        #initialize
        tokens = []
        labels = []
                                      
        sent_tokens = []
        sent_labels = []
        first_word = 1                                              #is the first word of a sentence
        for l,line in enumerate(lines):                
            if(line.isspace() == False):                            #not empty line   
                line=line.strip('\n') 
                token, pos = line.split('\t')
                sent_tokens.append(token)
                sent_labels.append(pos)

                if(first_word == 1):                                #the first word of a sentence
                    first_word = 0
                    
            elif((first_word == 0)):                                #end of sentence   
                tokens.append(sent_tokens)
                labels.append(sent_labels)
                sent_tokens = []
                sent_labels = []
                first_word = 1
            if(sent_tokens and (l == len(lines)-1)):
                tokens.append(sent_tokens)
                labels.append(sent_labels)
        return tokens, labels

def labels_to_index_map(all_training_labels):
    dict_ = {}
    c = 0
    for sent in all_training_labels:
        for label in sent:
            if label not in dict_:
                dict_[label] = c
                c+=1
    return dict_

def words2indexes(seq_of_words, w2i_lookup):
    """
    This function converts our sentence into a sequence of indexes that correspond to the rows in our embedding matrix
    :param seq_of_words: the document as a <list> of words
    :param w2i_lookup: the lookup table of {word:index} that we built earlier
    """
    seq_of_idxs = []
    for w in seq_of_words:
        i = w2i_lookup.get(w, 0) # we use the .get() method to allow for default return value if the word is not found
                                 # we've reserved the 0th row of embedding matrix for out-of-vocabulary words
        seq_of_idxs.append(i)
    return seq_of_idxs


def forward_pass(x):
    """
    This function will wrap all the steps needed to feed one sentence through the RNN
    :param x: a <list> of indices
    """
    # convert sequence of ints to sequence of embeddings
    input_seq = [embedding_parameters[i] for i in x]   # embedding_parameters can be used like <dict>
    # convert Parameters to Expressions
    W = dy.parameter(pW)
    b = dy.parameter(pb)
    # initialize the RNN unit
    rnn_seq = RNN_unit.initial_state()
    # run each timestep(word) through the RNN
    rnn_hidden_outs = rnn_seq.transduce(input_seq)

    # project each timestep's hidden output to size of labels
    rnn_outputs = [dy.transpose(W) * h + b for h in rnn_hidden_outs]
    return rnn_outputs

def train():

    # i = epoch index
    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)

    for i in range(num_epochs):
        random.seed(i+100)
        random.shuffle(train_tokens) 
        random.seed(i+100)
        random.shuffle(train_labels) 
        for j in range(num_batches_training):
            # begin a clean computational graph
            dy.renew_cg()
            # build the batch
            batch_tokens = train_tokens[j*batch_size:(j+1)*batch_size]
            batch_labels = train_labels[j*batch_size:(j+1)*batch_size]
            # iterate through the batch
            for k in range(len(batch_tokens)):
                # prepare input: words to indexes
                seq_of_idxs = words2indexes(batch_tokens[k], w2i)
                # make a forward pass
                preds = forward_pass(seq_of_idxs)
                # calculate loss for each token in each example
                loss = [dy.pickneglogsoftmax(preds[l], batch_labels[k][l]) for l in range(len(preds))]
                # sum the loss for each token
                sent_loss = dy.esum(loss)
                # backpropogate the loss for the sentence
                sent_loss.backward()
                trainer.update()


starttime = datetime.datetime.now()
train_tokens, train_labels = import_data('train.tagged')
flat_train_tokens = []
for token in train_tokens:
    flat_train_tokens.extend(token)
l2i = labels_to_index_map(train_labels)
train_labels = [[l2i[l] for l in sent] for sent in train_labels]
i2l = dict((v,k) for k,v in l2i.items())

#initialize empty model
RNN_model = dy.ParameterCollection()    

################
# HYPERPARAMETER
################
hidden_size = 200
# number of layers in `RNN`
num_layers = 1

#pretrained embeddings
emb_matrix_pretrained, w2i = load_pretrained_embeddings("glove.6B.200d.txt", take=10000)
embedding_dim = emb_matrix_pretrained.shape[1]
embedding_parameters = RNN_model.lookup_parameters_from_numpy(emb_matrix_pretrained)

#add RNN unit
RNN_unit = dy.VanillaLSTMBuilder(num_layers, embedding_dim, hidden_size, RNN_model)

#add projection layer
# W (hidden x num_labels) 
pW = RNN_model.add_parameters(
        (hidden_size, len(list(l2i.keys())))
)
dy.parameter(pW).npvalue().shape

# b (1 x num_labels)
pb = RNN_model.add_parameters(
        (len(list(l2i.keys())))        
)
# note: we are just giving one dimension (ignoring the "1" dimension)
# this makes manipulating the shapes in forward_pass() easier 
dy.parameter(pb).npvalue().shape


################
# HYPERPARAMETER
################
trainer = dy.SimpleSGDTrainer(
    m=RNN_model,
    # learning_rate=0.01
    learning_rate=0.008
)
batch_size = 256
num_epochs = 20

num_batches_training = int(np.ceil(len(train_tokens) / batch_size))
train()

RNN_model.save("trained.model")
#save tables
tables = []
tables.append(emb_matrix_pretrained) 
tables.append(flat_train_tokens)
tables.append(w2i)
tables.append(l2i)
tables.append(i2l)

with open('tables.txt', "wb") as f:
    pickle.dump(tables, f, pickle.HIGHEST_PROTOCOL)

endtime = datetime.datetime.now()
seconds = (endtime - starttime).seconds
minutes = (seconds % 3600) // 60
seconds = seconds % 60
print("Train: minutes:"+str(minutes)+" seconds:"+str(seconds))