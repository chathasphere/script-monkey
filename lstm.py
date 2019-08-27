import torch
import torch.nn as nn

class LSTM(nn.module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        :
