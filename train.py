import numpy as np
import sys
from helpers import one_hot, decode_one_hot
from random import seed, shuffle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from model import patched_forward_impl, CharRNN
import pdb


def extract_shakespeare_data():
    with open('data/t8.shakespeare.txt') as f:
        text = f.read()  

    text_lines = []
    #start after the header
    skip = False
    for line in text.split("\n")[244:]:
        if line[:2] == "<<":
            skip = True
        elif line[-2:] == ">>":
            skip = False
        if skip or line == "":
            continue
        text_lines.append(line)

    return text_lines


def encode_chars(text_lines, max_length):
    #data is a list of strings 
    #mythical double list (in)comprehension
    flatten = lambda l: [item for sublist in l for item in sublist]
    flattened = flatten(text_lines)

    chars = tuple(set(flattened))
    int2char = dict(enumerate(chars))
    char2int = {value: key for key, value in int2char.items()}

    numeric_sequences = [[char2int[char] for char in line][:max_length] for line in text_lines]

    return numeric_sequences, len(chars)


def main():

    #think about how I can modularize custom extraction of text
    #data is a list of strings (sentences, lines)
    data = extract_shakespeare_data()[:1000]

    sequences, n_chars = encode_chars(data, 70)
    
    # randomly shuffle the sequences
    shuffle(sequences)
    
    # split into test and validation set 
    n_training_sequences = int(.9 * len(sequences))
    training = sequences[:n_training_sequences]
    validation = sequences[n_training_sequences:]

    #numer of neurons in hidden layer of LSTM
    hidden_size = 100
    epochs = 1
    batch_size = 20
    lr = 0.01
    evaluate_per = 1
    rnn = CharRNN(n_chars, hidden_size)
    #needed for dropout: sets neural network to training mode
    #will be deactivated for the eval loop
    rnn.train()

    optimizer = torch.optim.Adam(rnn.parameters(), lr = lr)
    #this is equivalent to a (log) softmax activation layer + negative log likelihood
    loss_function = nn.CrossEntropyLoss()
    #double check that sequence length isn't overshooting the longest sequence
    
    #TODO need to understand what's going on with the "device" variable
    #particularly when transferring this to a GPU
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    for e in range(epochs):
        n_sequences = len(training)
        hx = rnn.init_hidden(batch_size)
        
        #loop through minibatches
        for i in range(0, n_sequences, batch_size): 

            batch = training[i:i+batch_size]
            batch = sorted(batch, key = lambda x: len(x), reverse=True)
            
            input_sequences, target_sequences = [], []
            sequence_lengths = []
            #get input and target sequences, one-hot encode them.
            #get sequence lengths too
            for sequence in batch:
                encoded = one_hot(sequence, n_chars)
                input_sequences.append(encoded[:-1])
                target_sequences.append(encoded[1:])
                sequence_lengths.append(len(encoded) - 1) #careful with the offsetting ;)
            
            #move this to the forward function of CharRnn?
            padded_input = pad_sequence(input_sequences)
            packed_input = pack_padded_sequence(padded_input, sequence_lengths)

            output, hx = rnn(packed_input, hx)

            #what is loss? 
            #loss = loss_function(output, target_sequences with some reshaping?
            #loss.backward()

            rnn.zero_grad()

            if (e + 1) % evaluate_per == 0:
                rnn.eval()
                pass
                #TODO
                #i need some batching for the eval process too, it would seem. can I 
                #generalize it into a function?
                rnn.train()
                print(f"epoch: {e+1}/{epochs}") # think of more useful print statements

    
    #validate_packing(packed_batches, int2char)

    test_out =  shakespeare_net(packed_batch, hx)




if __name__ == "__main__":
    seed(1609)
    main()
