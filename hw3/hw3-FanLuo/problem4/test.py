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

def predict(list_of_outputs):
    """
    This function will convert the outputs from forward_pass() to a <list> of label indexes
    """
    # take the softmax of each timestep
    # note: this step isn't actually necessary as the argmax of the raw outputs will come out the same
    # but the softmax is more "interpretable" if needed for debugging
    pred_probs = [dy.softmax(o) for o in list_of_outputs]  

    # convert each timestep's output to a numpy array
    pred_probs_np = [o.npvalue() for o in pred_probs]
    # take the argmax for each step
    pred_probs_idx = [np.argmax(o) for o in pred_probs_np]
    return pred_probs_idx

def test():
    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)
    all_predictions = []

    for j in range(num_batches_testing):
        # begin a clean computational graph
        dy.renew_cg()
        # build the batch
        batch_tokens = test_tokens[j*batch_size:(j+1)*batch_size]
        batch_labels = test_tokens[j*batch_size:(j+1)*batch_size]
        # iterate through the batch
        for k in range(len(batch_tokens)):
            # prepare input: words to indexes
            seq_of_idxs = words2indexes(batch_tokens[k], w2i)
            # make a forward pass
            preds = forward_pass(seq_of_idxs)
            label_preds = predict(preds)
            all_predictions.append(label_preds)
    return all_predictions

def get_accuracy(flat_list_of_scores):
    return float(sum(flat_list_of_scores) / len(flat_list_of_scores))

def check_score(pred, true_y):
    return 1 if pred == true_y else 0

def evaluate(nested_preds, nested_true, indexes):
    flat_scores = []
    flat_indexed_scores = [] 
    for i in range(len(nested_true)):
        scores = []
        pred = nested_preds[i]
        true = nested_true[i]
        for p,t in zip(pred,true):
            score = check_score(p,t)
            scores.append(score)
        flat_scores.extend(scores)

    for i, score in enumerate(flat_scores):
        if(i in indexes):
           flat_indexed_scores.append(score)

    overall_accuracy = get_accuracy(flat_scores) 
    indexed_accuracy = get_accuracy(flat_indexed_scores) 
    return overall_accuracy,indexed_accuracy
        
 

starttime = datetime.datetime.now()

#load tables
with open("tables.txt", 'rb') as model:     
    content = pickle.load(model)
    emb_matrix_pretrained = content[0]
    flat_train_tokens = content[1]
    w2i = content[2]
    l2i = content[3]  
    i2l = content[4]   

# test_tokens, test_labels = import_data('test.tagged')
test_tokens, test_labels = import_data('dev.tagged')
test_labels = [[l2i[l] for l in sent] for sent in test_labels]
unknown_index = []
index = 0
for i in range(len(test_tokens)):
    for test_token in test_tokens[i]:
        if(test_token not in flat_train_tokens):
           unknown_index.append(index) 
        index += 1
print(index)
print(len(unknown_index))

#initialize empty model
RNN_model = dy.ParameterCollection()    

################
# HYPERPARAMETER
################
hidden_size = 200
# number of layers in `RNN`
num_layers = 1

#pretrained embeddings
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

RNN_model.populate("trained.model")

batch_size = 256
num_batches_testing = int(np.ceil(len(test_tokens) / batch_size))
predictions = test()
overall_accuracy, unknown_accuracy = evaluate(predictions, test_labels, unknown_index)
endtime = datetime.datetime.now()
seconds = (endtime - starttime).seconds
minutes = (seconds % 3600) // 60
seconds = seconds % 60
print("Test: minutes:"+str(minutes)+" seconds:"+str(seconds))
print("Overall Accuracy: %.2f%%\n" % (100*overall_accuracy))
print("Unkonwn Accuracy: %.2f%%\n" % (100*unknown_accuracy))

