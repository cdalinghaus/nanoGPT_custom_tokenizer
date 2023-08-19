from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

# Sample 1 Million lines of text from the (potentially huge) input.txt file
# in hopes of a representative sample
os.system("rm sampled_file.txt")
os.system("shuf -n 1000000 input.txt > sampled_file.txt")
os.system("rm bpe_tokenizer.json")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
#tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=30000, show_progress=True)
tokenizer.train(files=["sampled_file.txt"], trainer=trainer)

# Save the tokenizer for later use
tokenizer.save("bpe_tokenizer.json")
os.system("rm sampled_file.txt")
