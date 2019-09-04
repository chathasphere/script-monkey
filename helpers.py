import torch
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial

def one_hot(sequence, n_states):
    """
    Given a list of integers and the maximal number of unique values found
    in the list, return a one-hot encoded tensor of shape (m, n)
    where m is sequence length and n is n_states.
    """
    if torch.cuda.is_available():
        return torch.eye(n_states)[sequence,:].cuda()
    else:
        return torch.eye(n_states)[sequence,:]


def decode_one_hot(vector):
    return vector.nonzero().item()

def prepare_batches(sequences, batch_size):
    n_sequences = len(sequences)
    for i in range(0, n_sequences, batch_size):
        batch = sequences[i:i+batch_size]
        input_sequences, target_sequences = [], []

        for sequence in batch:
            input_sequences.append(sequence[:-1])
            target_sequences.append(sequence[1:])

        yield input_sequences, target_sequences

def get_target_tensor(target_sequences):

    target_tensors = [torch.tensor(s) for s in target_sequences]
    
    if torch.cuda.is_available():
        return torch.stack(target_tensors).flatten().cuda()
    else:
        return torch.stack(target_tensors).flatten()

def make_sequences(text, sequence_length=100):
    """
    Split a text into sequences of the same length in characters.
    """
    n_sequences = len(text) // sequence_length
    sequences = []
    for i in range(0, n_sequences):
        sequence = text[i*sequence_length : (i+1)*sequence_length]
        sequences.append(sequence)

    return sequences

def sample(model, encoder, size, prime_str, temperature):

    model.eval()
    output_str = prime_str
    input_sequence = [encoder.char2int[char] for char in prime_str]

    hx = None
    
    for i in range(size):

        out, hx = model([input_sequence], hx)
        
        hx = tuple(h.detach() for h in hx)

        out = out.squeeze()
        
        dist = (F.log_softmax(out, dim=-1).data / temperature).exp()

        probs = dist.t() / dist.sum(dim=-1)
        
        if len(probs.shape) == 1:
          probs = probs.unsqueeze(1)

        #TODO use topk?
        # torch.zeros(2, 5).scatter_(indices, topk) would be a place to start methinks
       
        next_char_ix = torch.multinomial(probs[:,-1],1).item()

        input_sequence = [next_char_ix]

        output_str += encoder.int2char[next_char_ix]

    return output_str


