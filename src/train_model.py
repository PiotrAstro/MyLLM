import math
import time
import numpy as np
import torch
import pathlib
import tokenization
import transformer
import json

TOKENIZED_TRAINING_DATA_PATH = pathlib.Path("data", "dataopenwebtext", "tokenized_dataopenwebtext_32767.npy")
TRAINED_TOKENIZER_PATH = pathlib.Path("results", "tokenizer_dataopenwebtext_32767.txt")
MODEL_SAVE_DIR = pathlib.Path("results", "model_32768_tokens")

TOKENIZER = tokenization.BPE_Tokenizer()
TOKENIZER.load(TRAINED_TOKENIZER_PATH)

SEQUENCE_LENGTH = 512
GPT_CONFIG = {
    "max_sequence_length": SEQUENCE_LENGTH,
    "embeding_size": 512,
    "feed_forward_hidden_n": 512 * 4,
    "tokens_number": TOKENIZER.get_tokens_number(),
    "num_heads": 8,
    "blocks_n": 8,
    "residual_dropout": 0.1,
    "attention_dropout": 0.1,
    "embeding_dropout": 0.1
}

FLOAT_MODE = torch.float32  # torch.float64 or torch.float32 or torch.float16 or torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 24
WEIGHT_DECAY = 0.01
LEARNING_RATE = 1e-4
SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
SCHEDULER_KWARGS = {
    "T_0": 5,  # Initial restart period
    "T_mult": 2,  # Multiply period by this factor after each restart
    "eta_min": 1e-6  # Minimum learning rate
}
INPUT_STRIDE = SEQUENCE_LENGTH // 2
EPOCHS_N = 30
SAVE_MODEL_EACH_N_EPOCHS = 10

def get_model_save_path(epoch: int) -> pathlib.Path:
    return pathlib.Path(MODEL_SAVE_DIR, f"epoch_{epoch:03d}.dat")

def train_routine():
    # save gpt config
    with open(pathlib.Path(MODEL_SAVE_DIR, "config.json"), 'w') as f:
        json.dump(GPT_CONFIG, f)

    log_header = "epoch, time, learning_rate, avg_perplexity"
    with open(pathlib.Path(MODEL_SAVE_DIR, "log.log"), 'w') as f:
        f.write(log_header + "\n")

    gpt = transformer.MyTransformer(**GPT_CONFIG).to(FLOAT_MODE).to(DEVICE)
    gpt.save(get_model_save_path(0))

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Using {DEVICE} device")
    print(f"Model has {gpt.parameters_number():_} parameters")
    print("Preparing batches")

    loaded_np_tokens = np.load(TOKENIZED_TRAINING_DATA_PATH, allow_pickle=False)
    loaded_torch_tokens = torch.tensor(loaded_np_tokens, dtype=torch.long, requires_grad=False).to(DEVICE)

    sequences = []
    max_start_index = len(loaded_np_tokens) - SEQUENCE_LENGTH - 1
    for start_index in range(0, max_start_index, INPUT_STRIDE):
        input_seq = loaded_torch_tokens[start_index:start_index + SEQUENCE_LENGTH]
        output_seq = loaded_torch_tokens[start_index + 1:start_index + SEQUENCE_LENGTH + 1]
        sequences.append((input_seq, output_seq))

    data_loader = torch.utils.data.DataLoader(sequences, batch_size=BATCH_SIZE, shuffle=True)  # type: ignore
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = SCHEDULER(
        optimizer,
        **SCHEDULER_KWARGS
    )
    
    print("Start training")

    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCHS_N):
        time_start = time.time()
        perplexity_sum = 0
        batches_performed = 0
        for (input_batch, output_batch) in data_loader:
            optimizer.zero_grad()

            predicted = gpt(input_batch)
            predicted_for_loss = predicted.view(-1, predicted.size(-1))
            output_batch_for_loss = output_batch.view(-1)

            loss_here = loss(predicted_for_loss, output_batch_for_loss)
            perplexity_here = math.exp(loss_here.item())
            perplexity_sum += perplexity_here
            batches_performed += 1

            loss_here.backward()
            optimizer.step()

            print("\033[s", end="", flush=True)  # Save cursor position
            print(f"{perplexity_here:.2f}\033[K", end="", flush=True)  # Print and clear rest of line
            print("\033[u", end="", flush=True)  # Restore cursor position

            if epoch % SAVE_MODEL_EACH_N_EPOCHS == 0:
                gpt.save(get_model_save_path(epoch))

        avg_perplexity = perplexity_sum / batches_performed
        scheduler.step()
        time_end = time.time()
        gpt.save(get_model_save_path(epoch))
        log_text = f"{epoch}, {time_end - time_start:.2f}, {scheduler.get_last_lr()[0]:.6f}, {avg_perplexity:.4f}"
        with open(pathlib.Path(MODEL_SAVE_DIR, "log.log"), 'a') as f:
            f.write(log_text + "\n")
        print(log_header + "\n" + log_text + "\n\n\n")

if __name__ == "__main__":
    train_routine()
