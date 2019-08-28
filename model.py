import torch.nn as nn
import torch
from torch.nn import _VF
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import pdb


def patched_forward_impl(self, input, hx, batch_sizes, 
    max_batch_size, sorted_indices):
    # type: (Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor], int, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
    if hx is None:
        num_directions = 2 if self.bidirectional else 1
        zeros = torch.zeros(self.num_layers * num_directions,
                            max_batch_size, self.hidden_size,
                            dtype=input.dtype, device=input.device)
        hx = (zeros, zeros)
    else:
        # Each batch of the hidden state should match the input sequence that
        # the user believes he/she is passing in.
        hx = self.permute_hidden(hx, sorted_indices)

    self.check_forward_args(input, hx, batch_sizes)
    if batch_sizes is None:
        result = _VF.lstm(input, hx, self._get_flat_weights(), self.bias, self.num_layers,
                          self.dropout, self.training, self.bidirectional, self.batch_first)
    else:
        result = _VF.lstm(input, batch_sizes, hx, tuple(self._get_flat_weights()), bool(self.bias),
                          self.num_layers, self.dropout, self.training, self.bidirectional)
    output = result[0]
    hidden = result[1:]

    return output, hidden


class CharRNN(nn.Module):
    
    def __init__(self, n_chars, hidden_size, n_rnn_layers=1, dropout=0):
        
        super().__init__()
        
        self.n_layers = n_rnn_layers
        self.n_hidden = hidden_size
        self.n_chars = n_chars
        #input size corresponds to the number of unique characters
        #sigh, this is bad practice
        nn.LSTM.forward_impl = patched_forward_impl
        self.lstm = nn.LSTM(n_chars, hidden_size, n_rnn_layers, dropout)
        
        #decoder layer?
        self.dense = nn.Linear(hidden_size, n_chars)
        
        
    def forward(self, seq, hx):
        
        recurrent_output, _ = self.lstm(seq, hx)
        
        X, sequence_lengths = pad_packed_sequence(recurrent_output)

        #X = X.view(-1, X.shape[2])

        linear_output  = self.dense(X)

        pdb.set_trace()

        
    
    def init_hidden(self, batch_size):
        
        weight0 = next(self.parameters()).data
        hidden = (weight0.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight0.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

