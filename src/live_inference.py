import time
import numpy as np
import torch
import pathlib
import tokenization
import transformer
import json

TRAINED_TOKENIZER_PATH = pathlib.Path("results", "tokenizer_dataopenwebtext_32767.txt")
MODEL_SAVE_PATH = pathlib.Path("results", "model1.dat")
CONFIG_SAVE_PATH = pathlib.Path("results", "model1.json")

with open(CONFIG_SAVE_PATH, 'r') as f:
    GPT_CONFIG = json.load(f)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOAT_MODE = torch.float32  # torch.float64 or torch.float32 or torch.float16 or torch.bfloat16
TEMPERATURE = 0.7
ONE_PROMPT_MAX_OUTPUT_TOKENS = 100


def live_inference(gpt: transformer.MyTransformer, tokenizer: tokenization.BPE_Tokenizer, temperature: float = TEMPERATURE, device: torch.device = DEVICE, one_prompt_max_output_tokens: int = ONE_PROMPT_MAX_OUTPUT_TOKENS):
    print(f"Model has {gpt.parameters_number():_} parameters")
    print(f"Temperature: {temperature}")
    print(f"One prompt max output tokens: {one_prompt_max_output_tokens}")
    print(f"Device: {device}")
    print(f"Float mode: {FLOAT_MODE}")
    print("\n\nEnter your text or write 'exit' to quit:")

    while True:
        print("\n>>> ", end="")
        user_input = input()
        input_tokens = torch.tensor([], dtype=torch.long, requires_grad=False).to(device).view(1, -1)
        if user_input == "exit":
            break
        else:
            # user_input += "<|endoftext|>"
            print("\nPjoterLLM >>> ", end="")
            user_input_tokens = tokenizer.encode(user_input)
            user_input_tensor = torch.tensor(user_input_tokens, dtype=torch.long, requires_grad=False).to(device).view(1, -1)
            input_tokens = torch.cat((input_tokens, user_input_tensor), dim=1)
            this_prompt_output_count = 0
            this_prompt_output_tokens = []

            with torch.no_grad():
                while True:
                    if this_prompt_output_count >= one_prompt_max_output_tokens:
                        this_prompt_output_tokens.append(tokenizer.encode("<|endoftext|>")[0])
                        break
                    this_prompt_output_count += 1

                    if input_tokens.size(1) > gpt.max_sequence_length:
                        input_tokens = input_tokens[:, -gpt.max_sequence_length:]

                    output = gpt(input_tokens)
                    output_tokens = output[0, -1, :]
                    if temperature > 0:
                        output_tokens = output_tokens / temperature
                        output_tokens = torch.softmax(output_tokens, dim=-1)
                        output_token = torch.multinomial(output_tokens, num_samples=1)
                    else:
                        output_token = torch.argmax(output_tokens, dim=-1)

                    input_tokens = torch.cat((input_tokens, output_token.view(1, -1)), dim=1)

                    output_token_value = int(output_token.item())
                    this_prompt_output_tokens.append(output_token_value)
                    decoded_output = tokenizer.decode([output_token_value])[0]
                    if decoded_output == "<|endoftext|>":
                        break
                    print("\033[s", end="", flush=True)  # Save cursor position
                    print(f"{tokenizer.decode(this_prompt_output_tokens)}\033[K", end="", flush=True)  # Print and clear rest of line
                    print("\033[u", end="", flush=True)  # Restore cursor position
            
            # print("\033[2J\033[H", end="", flush=True)  # Clear screen and move cursor to top
            print(f"{tokenizer.decode(this_prompt_output_tokens[:-1])}\033[K")

if __name__ == "__main__":
    gpt = transformer.MyTransformer(**GPT_CONFIG)
    gpt.load(MODEL_SAVE_PATH)
    gpt.to(FLOAT_MODE).to(DEVICE)

    tokenizer = tokenization.BPE_Tokenizer()
    tokenizer.load(TRAINED_TOKENIZER_PATH)
    live_inference(gpt, tokenizer)
