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

def prepare_batches(sequences, batch_size, n_states):
    
    n_sequences = len(sequences)
    for i in range(0, n_sequences, batch_size):
        batch = sequences[i:i+batch_size]
        batch = sorted(batch, key = lambda x: len(x), reverse=True)
        
        input_sequences, target_sequences = [], []
        for sequence in batch:
            encoded = one_hot(sequence, n_states)
            input_sequences.append(encoded[:-1])
            target_sequences.append(sequence[1:])

        yield input_sequences, target_sequences

def get_target_tensor(target_sequences, sequence_lengths):

    target_tensors = [torch.tensor(s) for s in target_sequences]
    padded_targets = pad_sequence(target_tensors)
    
    return pack_padded_sequence(padded_targets, sequence_lengths)

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
