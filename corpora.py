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

    def encode_sequences(self, text_sequences, max_length):
        #this may be called "vectorizing?

        return [[self.char2int[char] for char in sequence][:max_length] for sequence in text_sequences]


def extract_shakespeare_data(path = "data/t8.shakespeare.txt"):
    """
    Load the MIT online Shakespeare corpus from a text file.
    
    Args:
        path (str): path to Shakespare text file

    Returns:
        text (str): entire text
        corpus (list of str): list of words found in text, including duplicates
        lines (list of list of str): list of lines found in text
    """

    with open(path) as f:
        text = f.read()

    lines = []
    corpus = []
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
        corpus.extend(line)
        lines.append(line)

    return cleaned_text, corpus, lines

