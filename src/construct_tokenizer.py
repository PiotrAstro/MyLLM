import pathlib
import tokenization

TOKENS_N = 32_767  # additional one will be for eof

DATA_PATH = pathlib.Path("data", "dataopenwebtext", "train")
SAFE_PATH = pathlib.Path("results", f"tokenizer_dataopenwebtext_{TOKENS_N}.txt")

if __name__ == "__main__":
    tokenizer = tokenization.BPE_Tokenizer()
    tokenizer.construct(DATA_PATH, TOKENS_N)
    tokenizer.save(SAFE_PATH)