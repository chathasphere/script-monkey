import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pdb

def one_hot(sequence, n_states):
    """
    Given a list of integers and the maximal number of unique values found
    in the list, return a one-hot encoded tensor of shape (m, n)
    where m is sequence length and n is n_states.
    """
    return torch.eye(n_states)[sequence,:]


def decode_one_hot(vector):
    return vector.nonzero().item()


def prepare_batches(sequences, batch_size, n_states, sequence_length):
    """
    Given a list of numeric sequences, returns batches of input and target sequences.

    Target sequences are input sequences offset by one timestep. The sequence "hello world" would give 
    an input "hello worl" and a target "ello world".
    
    The target and input sequences get converted to one-hot encoded batches of the following shape:
        (sequence_length x batch_size x n_states)
    
      - n_states is the dimensionality of the one-hot encoding.

      - sequence_length is the length of the longest sequence. Shorter sequences are padded to that
    length. To avoid backpropagating over padded elements, the entire batch is "packed." Whatever that
    entails.

    Returns two Pytorch PackedSequence objects,  
    #do I need to pack the target sequence? probably not. 
    """
    sequences = sorted(sequences, key = lambda x: len(x), reverse=True)

    input_sequences 
    batches = []
    n_sequences = len(sequences)
    for i in range(0, n_sequences, batch_size):
        batches.append(sequences[i:i+batch_size])

    packed_batches = []
    for batch in batches:
        sequence_lengths = [len(s) for s in batch]
        coded_batch = [one_hot(s, n_states) for s in batch]
        padded_batch = pad_sequence(coded_batch)
        #dims should be: (seq_len, batch_size, n_chars)
        packed_batches.append(pack_padded_sequence(padded_batch, 
            sequence_lengths))

    return packed_batches

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
