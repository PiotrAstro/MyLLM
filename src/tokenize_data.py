import pathlib
import tokenization
import numpy as np

DATA_PATH = pathlib.Path("data", "dataopenwebtext", "train")
TOKENIZED_SAFE_PATH = pathlib.Path("data", "dataopenwebtext","tokenized_dataopenwebtext_32767.npy")
TOKENIZER_PATH = pathlib.Path("results", "tokenizer_dataopenwebtext_32767.txt")

if __name__ == "__main__":
    tokenizer = tokenization.BPE_Tokenizer()
    tokenizer.load(TOKENIZER_PATH)
    tokenized_data = tokenizer.load_and_encode(DATA_PATH)
    np.save(TOKENIZED_SAFE_PATH, tokenized_data)
