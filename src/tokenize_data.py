import pathlib
import tokenization
import numpy as np

DATA_PATH = pathlib.Path("data", "fineweb", "train")
TOKENIZED_SAFE_PATH = pathlib.Path("data", "tokenized_fineweb_32768.npy")
TOKENIZER_PATH = pathlib.Path("results", "tokenizer_fineweb_32768.txt")

if __name__ == "__main__":
    tokenizer = tokenization.BPE_Tokenizer()
    tokenizer.load(TOKENIZER_PATH)
    tokenized_data = tokenizer.load_and_encode(DATA_PATH)
    np.save(TOKENIZED_SAFE_PATH, tokenized_data)
