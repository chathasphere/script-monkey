import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
import torch.nn.functional as F
import pdb

class CharRNN(nn.Module):
    
    def __init__(self, n_chars, hidden_size, n_rnn_layers=1, dropout=0):
        
        super().__init__()
        
        self.n_layers = n_rnn_layers
        self.n_hidden = hidden_size
        self.n_chars = n_chars
        #input size corresponds to the number of unique characters
        self.lstm = nn.LSTM(n_chars, hidden_size, n_rnn_layers, dropout=dropout)
        
        #decoder layer?
        self.dense = nn.Linear(hidden_size, n_chars)
        
        
    def forward(self, sequences, hx, sequence_lengths):

        padded_sequences = pad_sequence(sequences)
        packed_sequences = pack_padded_sequence(padded_sequences, sequence_lengths)
        
        recurrent_output, hidden = self.lstm(packed_sequences, hx)

        linear_output  = self.dense(recurrent_output[0])

        return linear_output, hidden
        
    
    def init_hidden(self, batch_size):
        
        weight0 = next(self.parameters()).data
        hidden = (weight0.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight0.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

