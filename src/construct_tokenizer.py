import pathlib
import tokenization

TOKENS_N = 32_768

DATA_PATH = pathlib.Path("data", "fineweb", "train")
SAFE_PATH = pathlib.Path("results", f"tokenizer_fineweb_{TOKENS_N}.txt")

if __name__ == "__main__":
    tokenizer = tokenization.BPE_Tokenizer()
    tokenizer.construct(DATA_PATH, TOKENS_N - 1)  # additional one will be for eof
    tokenizer.save(SAFE_PATH)