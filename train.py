import numpy as np
import sys
from helpers import one_hot, prepare_batches, get_target_tensor
from random import seed, shuffle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from model import CharRNN
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
    #potentially want a way to hold onto int2char,char2int...
    #TODO convert this into a class

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
    epochs = 3
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
        training_batches = prepare_batches(training, batch_size, n_chars)
        for input_sequences, target_sequences in training_batches:
            #hx = rnn.init_hidden(batch_size)
            hx = None
            sequence_lengths = [len(sequence) for sequence in input_sequences]
            y_hat, hx = rnn(input_sequences, hx, sequence_lengths)
            
            #this should be more explicity
            #why are you just picking up the first element of the tuple?
            y  = get_target_tensor(target_sequences,
                    sequence_lengths)[0]
            #this is almost certainly wrong
            loss = loss_function(y_hat, y)
            loss.backward()
            #consider clipping grad norm
            optimizer.step()
            rnn.zero_grad()

        if (e + 1) % evaluate_per == 0:
            hx = rnn.init_hidden(batch_size)

            rnn.eval()
            validation_batches = prepare_batches(validation,
                    batch_size, n_chars)
            #get loss per batch
            val_losses = []
            n_batches = 0
            for input_sequences, target_sequences in validation_batches:

                sequence_lengths = [len(sequence) for sequence in input_sequences]
                y_hat, hx = rnn(input_sequences, hx, sequence_lengths)

                y = get_target_tensor(target_sequences,
                        sequence_lengths)[0]

                val_loss = loss_function(y_hat, y)
                val_losses.append(val_loss.item())
                n_batches += 1
            mean_val_loss = sum(val_losses) / n_batches

            rnn.train()
            print(f"epoch: {e+1}/{epochs}")
            print(f"training loss: {loss.item():.2f}")
            print(f"validation loss: {mean_val_loss:.2f}")
    
    #validate_packing(packed_batches, int2char)




if __name__ == "__main__":
    seed(1609)
    main()
