import os
import pickle
import requests
import numpy as np
import tempfile
import shutil
from tokenizers import Tokenizer
import gc

"""

Create the .bin files but in a chunked manner (since input.txt might be huge)

"""

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

os.system("rm train.bin")
os.system("rm val.bin")

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'combinedmay.txt')

def encode(s):
    return tokenizer.encode(s).ids  # This will output a list of integers

def decode(l):
    return tokenizer.decode(l)  # This will output a string

chunk_size = 500  # or whatever number of lines fits in memory

# Open the output file in binary append mode
with open('data.bin', 'ab') as bin_file:
    with open("combinedmay.txt", 'r') as f:
        while True:
            # read a chunk of the file
            lines = [f.readline() for _ in range(chunk_size)]

            # check if we've reached the end of the file
            if not lines or lines[0] == '':
                break

            # concatenate the lines and tokenize/encode the chunk
            chunk = ''.join(lines)
            encoded_chunk = encode(chunk)

            # convert to numpy array and write the encoded chunk to the .bin file
            np.array(encoded_chunk, dtype=np.uint16).tofile(bin_file)
            del chunk, encoded_chunk, lines
            gc.collect()

# Now split the binary file into training and validation sets
with open('data.bin', 'rb') as data_file, open('train.bin', 'wb') as train_file, open('val.bin', 'wb') as val_file:
    data_file.seek(0, os.SEEK_END)
    total_size = data_file.tell()

    # Make sure train_size is a multiple of 2
    train_size = int(total_size * 0.9)
    train_size -= train_size % 2  # this will subtract 1 from train_size if it's not a multiple of 2

    val_size = total_size - train_size  # no need to adjust val_size, since total_size and train_size are both multiples of 2

    data_file.seek(0, os.SEEK_SET)

    train_data = data_file.read(train_size)
    train_file.write(train_data)

    val_data = data_file.read(val_size)
    val_file.write(val_data)


