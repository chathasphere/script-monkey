# About
Inspired by the [Char-RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) of Andrej Karpathy, 
in this project I train a Pytorch LSTM (long short term memory) RNN (recurrent neural network) on text sequences in order to create parody 
text. The LSTM learns a probabilistic model of text sequences, from which characters are sampled to generate new text.

#### Why Script Monkey?
Because it's conjectured that given enough monkey-years and typewriters, it would be possible to replicate the complete works of Shakespeare.
The example I give trains on Shakespeare's complete works, though it's easily adaptable to other large (think a few MB) text corpuses.

## How it works
#### Text preprocessing
A corpus is split into sequences of a fixed length (say, 100 characters). Each sequence yields an *input sequence* and a *target sequence*, 
with the latter offset forward by one timestep. So the sequence "TO BE OR NOT TO BE" means an input "TO BE OR NOT TO B" and a target "O BE OR NOT TO BE".
Of course, we first have to vectorize text sequences by mapping unique characters to integers (about 80 in Shakespeare's works, including punctuation and
escape characters). For efficiency, sequences are grouped in batches of fixed size.
#### Training
Input sequences get one-hot encoded and fed into a Char-RNN neural network with the following architecture:
- One or more recurrent LSTM layers: represents sequences in a dimensionality defined by the number of neurons in the hidden layer)
- A densely connected linear layer that projects the recurrent output back down onto the space of unique characters
- The network's output is interpreted the unnormalized log probabilities of characters at the next timestep
- The loss function is cross-entropy loss (softmax + negative log likelihood)
Note: training is likely to be very slow unless run on a GPU. I recommend [Google Colab](https://colab.research.google.com/) for a free solution.
#### Sampling
Using the `sample` function in `helpers.py`, you can generate sample strings of a designated length given a "prime string" that initializes the RNN's hidden state (viz. the models long and short term memory) and a probability distribution of characters.
Each subsequent iteration randomly selects a character from this distribution and updates the hidden state/distribution.

### Suggested Reading
- https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
- https://arxiv.org/abs/1506.02078
- https://towardsdatascience.com/writing-like-shakespeare-with-machine-learning-in-pytorch-d77f851d910c
- https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally
- https://www.tensorflow.org/tutorials/sequences/text_generation
