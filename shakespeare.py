from corpora import CharEncoder, extract_shakespeare_data 
from random import seed, shuffle
from model import CharRNN
from train import train
from helpers import  make_sequences, sample
import argparse

def main():
    parser = argparse.ArgumentParser("Char-RNN on the complete works of Shakespeare")
    parser.add_argument("--test", type=bool, default=False,
            help = "if true, keep only a thousand lines from the Shakespeare corpus")

    args = parser.parse_args()

    seed(1616)
    
    text = extract_shakespeare_data("data/t8.shakespeare.txt")
    char_encoder = CharEncoder(text)
    #get sequences of 100 characters
    sequences = make_sequences(text)
    #vectorize with numeric labeling
    #each character gets mapped to an integer & vice versa
    sequences = char_encoder.label_sequences(sequences)
    if args.test:
        print("Test: downsizing data to 1,000 sequences...")
        sequences = sequences[:1000]

    shuffle(sequences)
    n_training_sequences = int(.9 * len(sequences))
    #split the dataset into training and validation sets
    training = sequences[:n_training_sequences]
    validation = sequences[n_training_sequences:]

    hidden_size = 128 
    rnn = CharRNN(char_encoder.n_chars, hidden_size)
    train(rnn, training, validation, epochs = 4, lr = 0.01, evaluate_per = 2, batch_size = 20)
    
    print(sample(rnn, prime_str = "Macbeth", size = 100, encoder = char_encoder,
            temperature=.9))


if __name__ == "__main__":
    main()
