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
    '''
    Given a one-hot encoded vector, return the non-zero index
    '''
    return vector.nonzero().item()

def prepare_batches(sequences, batch_size):
    """
    Splits a list of sequences into batches of a fixed size. Each sequence yields an input sequence
    and a target sequence, with the latter one time step ahead. For example, the sequence "to be or not
    to be" gives an input sequence of "to be or not to b" and a target sequence of "o be or not to be."
    """
    n_sequences = len(sequences)
    for i in range(0, n_sequences, batch_size):
        batch = sequences[i:i+batch_size]
        input_sequences, target_sequences = [], []

        for sequence in batch:
            input_sequences.append(sequence[:-1])
            target_sequences.append(sequence[1:])

        yield input_sequences, target_sequences

def get_target_tensor(target_sequences):
    """
    Flattens a batch of target sequences into one long tensor (length: number_sequences * sequence_length)
    """

    target_tensors = [torch.tensor(s) for s in target_sequences]
    
    if torch.cuda.is_available():
        return torch.stack(target_tensors).flatten().cuda()
    else:
        return torch.stack(target_tensors).flatten()

def make_sequences(text, sequence_length=100):
    """
    Split a text into sequences of the same length
    """
    n_sequences = len(text) // sequence_length
    sequences = []
    for i in range(0, n_sequences):
        sequence = text[i*sequence_length : (i+1)*sequence_length]
        sequences.append(sequence)

    return sequences

def sample(model, encoder, size, prime_str, temperature, topk=None):
    """
    Randomly generate text from a trained model.

    Args:
        model (nn.module object): trained LSTM neural network
        encoder (CharEncoder object): contains character-to-integer mappings
        size (int): length of generated sample
        prime_str (str): input to initialize the network's hidden state
        temperature (float in range (0,1]): dampens the model's character probability distribution:
            values less than 1 make the model more conservative, giving additional weight to
            high-probability guesses and penalizing low-probability guesses. As temperature goes to 
            zero, softmax becomes "argmax," and the most probable character is picked almost certainly.

    Returns:
        output_str (str): Randomly-generated text sample of length (size)
    """
    #deactivate training mode
    model.eval()
    #initialize output string
    output_str = prime_str
    #vectorize input string as a sequence of ints
    input_sequence = [encoder.char2int[char] for char in prime_str]
    #initialize hidden state to None
    hx = None
    for i in range(size): #generate characters of output string one at a time
        #get model output and hidden state (short-term and long-term memory)
        out, hx = model([input_sequence], hx)
        hx = tuple(h.detach() for h in hx)
        #ignore batch dimension because our batch is of size 1
        out = out.squeeze()
        #interpreting output as unnormalized logits, obtain probabilities of the next character, scaled
        #by temperature, conditioned on the input sequence.
        #a higher temperature means a softer probability distribution, i.e. less conservative predictions.
        probs = F.softmax(out/ temperature, dim=-1)
          
        #If probs are generated on a string of multiple characters,
        #keep prediction of next character only
        if len(probs.shape) > 1:
            probs = probs[-1,:]
        
        if topk is not None:
            #sample from only the top k most probable characters
            values, indices = probs.topk(topk)
            if torch.cuda.is_available():
                zeros = torch.zeros(encoder.n_chars).cuda()
            else:
                zeros = torch.zeros(encoder.n_chars)
            probs =  torch.scatter(zeros, 0, indices, values)  
        #sample a random character from the probability distribution
        next_char_ix = torch.multinomial(probs,1).item()
        #set the new input sequence as just the next predicted character while retaining hidden state
        input_sequence = [next_char_ix]
        #add the next character to the output string
        output_str += encoder.int2char[next_char_ix]

    return output_str

