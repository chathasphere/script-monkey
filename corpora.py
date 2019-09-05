"""Contains classes to extract text data and to encode text corpora"""
import pdb

class CharEncoder():
    """
    Contains data on an encoded text corpus with labelled characters, including unique characters and 
    mappings to/from characters to integers.
    """
    def __init__(self, corpus):
        """
        Args:
            corpus (list of str): a list containing every word in the text, including duplicates
        """
        self.chars = tuple(set(corpus))
        self.n_chars = len(self.chars)
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {value: key for key, value in self.int2char.items()}

    def label_sequences(self, text_sequences):
        #this may be called "vectorizing?

        return [[self.char2int[char] for char in sequence] for sequence in text_sequences]


def extract_shakespeare_data(path = "data/t8.shakespeare.txt"):
    """
    Load the MIT online Shakespeare corpus from a text file.
    
    Args:
        path (str): path to Shakespare text file

    Returns:
        cleaned_text (str): entire cleaned text stripped of header/notes
    """

    with open(path) as f:
        text = f.read()

    cleaned_text = ""
    skip = False
    
    for line in text.split("\n")[244:-1]:
        if line[:2] == "<<":
            skip = True
        elif line[-2:] == ">>":
            skip = False
            continue
        if skip or line == "":
            continue
        line = line+"\n"

        cleaned_text += line

    return cleaned_text

def extract_kjv_data(path = "data/kjv.txt"):
    """
    Load the King James Version of the Bible.
    """
    with open(path) as f:
        text = f.read()

    text = text[996:-18730]

    return text


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


    

