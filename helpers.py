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

#def predict_char(model, char_in, hx, encoder, temperature = 1):
#
#    ix = encoder.char2int[char_in]
#    out, hx = model([[ix]], hx, sequence_lengths=[1])
#    #the lower the temperature, the more conservative the model
#    #the higher it is (closer to one) the more confident it is
#    out = out / temperature
#    p = F.log_softmax(out, dim=1).data
#    top_chars = np.arange(encoder.n_chars)
#    #or restrict to just the top characters?
#
#    p = p.numpy().squeeze()
#
#    #or do I do a multinomial distribution to sample?
#    char_out = np.random.choice(top_chars, p = (p / p.sum()))
#
#    return encoder.int2char[char_out], h

def generate(model, prime_str, encoder, pred_length, temperature = 0.8):

    model.eval()
    #convert priming string into a one-hot encoded tensor
    prime_sequence = [encoder.char2int[char] for char in prime_str]

    output_str = prime_str
    hx = None
    for i in range(pred_length):
        #wrap things in lists because model expects batches
        out, hx = model([prime_sequence], hx, [len(output_str)])

        out = out / temperature
        ls = F.log_softmax(out, dim=1).data.squeeze()
        
        probs = ls.t() / ls.sum(dim=1)
        #get prediction of next character
        m = Multinomial(probs = probs[:,-1])
        next_char_ix = decode_one_hot(m.sample())

        prime_sequence.append(next_char_ix)
        output_str += encoder.int2char[next_char_ix]

    return output_str

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

