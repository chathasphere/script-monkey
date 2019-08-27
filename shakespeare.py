import numpy as np
import sys
from helpers import one_hot, prepare_batches, decode_one_hot
from random import seed, shuffle
from torch.nn.utils.rnn import pad_packed_sequence
from rnn_model import patched_forward_impl, TextGenRNN
import pdb


def extract_data():
    with open('data/t8.shakespeare.txt') as f:
        text = f.read()  

    shakespeare = []
    #start after the header
    skip = False
    for line in text.split("\n")[244:]:
        if line[:2] == "<<":
            skip = True
        elif line[-2:] == ">>":
            skip = False
        if skip or line == "":
            continue
        shakespeare.append(line)

    return shakespeare

def validate_packing(packed_batches, int2char):
			
    lines = []
    
    for packed_batch in packed_batches:
        
        unpacked_sequences, sequence_lengths = pad_packed_sequence(packed_batch)
        
        for i in range(len(sequence_lengths)):
            
            length = sequence_lengths[i]
            sequence = unpacked_sequences[:,i,:][:length]
            
            numbers_sequence = [decode_one_hot(vec) for vec in sequence]
            
            lines.append([int2char[num] for num in numbers_sequence])
    
    
    for i in range(15):
        print(''.join(lines[i]) + '\n')

def main():
    data = extract_data()[:1000]

    flatten = lambda l: [item for sublist in l for item in sublist]
    flattened = flatten(data)
    
    chars = tuple(set(flattened))
    int2char = dict(enumerate(chars))
    char2int = {value: key for key, value in int2char.items()}
    
    numeric_sequences = [[char2int[char] for char in line][:70] for line in data]
    
    # randomly shuffle the sequences
    shuffle(numeric_sequences)
    
    # split into test and validation set 
    n_training_sequences = int(.9 * len(numeric_sequences))
    training = numeric_sequences[:n_training_sequences]
    validation = numeric_sequences[n_training_sequences:]

    training_input = [sequence[:-1] for sequence in training]
    training_target = [sequence[1:] for sequence in training]

    packed_batches = prepare_batches(training_input, batch_size = 20,
       	n_states = len(chars), sequence_length = 69)

    
    #validate_packing(packed_batches, int2char)

    packed_batch = packed_batches[0]


    shakespeare_net = TextGenRNN(69,100)

    hx = shakespeare_net.init_hidden(20)
    shakespeare_net(packed_batch, hx)



if __name__ == "__main__":
    seed(1609)
    main()
