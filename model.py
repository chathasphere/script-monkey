import torch.nn as nn
import torch.nn.functional as F
from helpers import one_hot
from torch import stack

class CharRNN(nn.Module):
    
    def __init__(self, n_chars, hidden_size, n_rnn_layers=1, dropout=0):
        
        super().__init__()
        
        self.n_layers = n_rnn_layers
        self.n_hidden = hidden_size
        self.n_chars = n_chars
        #input size corresponds to the number of unique characters
        self.lstm = nn.LSTM(n_chars, hidden_size, n_rnn_layers, dropout=dropout, 
                batch_first=True)
        #decoder layer
        self.dense = nn.Linear(hidden_size, n_chars)
        
        
    def forward(self, input_sequences, hx):

        #one hot encode a list of sequences
        encoded_sequences = [one_hot(sequence, self.n_chars) 
                for sequence in input_sequences]
        
        #batch has dimensions (n_sequences x batch_size x n_chars)
        #fixinf a dimension
        batch = stack(encoded_sequences, dim=0)
        #pass into the LSTM
        recurrent_output, hidden = self.lstm(batch, hx)
        #"dense" layer just projects it back down to the space of available characters
        linear_output  = self.dense(recurrent_output)

        return linear_output, hidden
        
