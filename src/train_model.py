import time
import numpy as np
import torch
import pathlib
import tokenization
import transformer
import json

TOKENIZED_TRAINING_DATA_PATH = pathlib.Path("data", "dataopenwebtext", "tokenized_dataopenwebtext_32768.npy")
TRAINED_TOKENIZER_PATH = pathlib.Path("results", "tokenizer_dataopenwebtext_32768.txt")
MODEL_SAVE_PATH = pathlib.Path("results", "model1.dat")
CONFIG_SAVE_PATH = pathlib.Path("results", "model1.json")

TOKENIZER = tokenization.BPE_Tokenizer()
TOKENIZER.load(TRAINED_TOKENIZER_PATH)

SEQUENCE_LENGTH = 512
GPT_CONFIG = {
    "max_sequence_length": SEQUENCE_LENGTH,
    "embeding_size": 512,
    "feed_forward_hidden_n": 512 * 2,
    "tokens_number": TOKENIZER.get_tokens_number(),
    "num_heads": 8,
    "blocks_n": 8,
    "residual_dropout": 0.1,
    "attention_dropout": 0.1,
    "embeding_dropout": 0.1
}

FLOAT_MODE = torch.float32  # torch.float64 or torch.float32 or torch.float16 or torch.bfloat16
BATCH_SIZE = 12
LEARNING_RATE = 0.000_01
INPUT_STRIDE = SEQUENCE_LENGTH // 2
EPOCHS_N = 10
LOSS_TYPE = torch.nn.CrossEntropyLoss


def train_routine():
        # save gpt config
    with open(CONFIG_SAVE_PATH, 'w') as f:
        json.dump(GPT_CONFIG, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt = transformer.MyTransformer(**GPT_CONFIG).to(FLOAT_MODE).to(device)
    gpt.save(MODEL_SAVE_PATH)

    print(f"Model has {gpt.parameters_number():_} parameters")

    loaded_np_tokens = np.load(TOKENIZED_TRAINING_DATA_PATH, allow_pickle=False)
    loaded_torch_tokens = torch.tensor(loaded_np_tokens, dtype=torch.long, requires_grad=False).to(device)

    sequences = []
    max_start_index = len(loaded_np_tokens) - SEQUENCE_LENGTH - 1
    for start_index in range(0, max_start_index, INPUT_STRIDE):
        # Get input sequence view
        input_seq = loaded_torch_tokens[start_index:start_index + SEQUENCE_LENGTH]
        # Get output sequence view (shifted by 1)
        output_seq = loaded_torch_tokens[start_index + 1:start_index + SEQUENCE_LENGTH + 1]
        
        # Add tuple of (input, output) to the list
        sequences.append((input_seq, output_seq))

    data_loader = torch.utils.data.DataLoader(sequences, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(gpt.parameters(), lr=LEARNING_RATE)

    loss = LOSS_TYPE()
    for epoch in range(EPOCHS_N):
        time_start = time.time()
        for (input_batch, output_batch) in data_loader:
            predicted = gpt(input_batch)
            predicted_for_loss = predicted.view(-1, predicted.size(-1))
            output_batch_for_loss = output_batch.view(-1)
            loss_here = loss(predicted_for_loss, output_batch_for_loss)
            optimizer.step()
            optimizer.zero_grad()
        time_end = time.time()
        gpt.save(MODEL_SAVE_PATH)
        print(f"Calculated {epoch} epoch, time: {time_end - time_start:.2f}")


if __name__ == "__main__":
    train_routine()
