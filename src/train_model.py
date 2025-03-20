import math
import time
import numpy as np
import torch
import pathlib
import tokenization
import transformer
import json
from torch import autocast, GradScaler



TOKENIZED_TRAINING_DATA_PATH = pathlib.Path("data", "tokenized_fineweb_32768.npy")
TOKENIZED_VALIDATION_DATA_PATH = pathlib.Path("data", "tokenized_fineweb_32768_validation.npy")
TRAINED_TOKENIZER_PATH = pathlib.Path("results", "tokenizer_fineweb_32768.txt")
MODEL_SAVE_DIR = pathlib.Path("results", "model_fineweb_32768_tokens")

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
    "residual_dropout": 0.05,
    "attention_dropout": 0.05,
    "embeding_dropout": 0.05
}

FLOAT_AUTOCAST_MODE = torch.bfloat16  # torch.float64 or torch.float32 or torch.float16 or torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
VIRTUAL_BATCH_SIZE = 512  # Effective batch size we want to mimic
ACCUMULATION_STEPS = VIRTUAL_BATCH_SIZE // BATCH_SIZE  # How many batches to accumulate
WEIGHT_DECAY = 0.005
LEARNING_RATE = 1e-3
SCHEDULER = torch.optim.lr_scheduler.OneCycleLR
SCHEDULER_KWARGS = {
    "max_lr": 1e-3,             # Peak learning rate
    "pct_start": 0.1,            # Percentage of training spent in the increasing phase
    "div_factor": 25.0,            # initial_lr = max_lr/div_factor
    "final_div_factor": 100.0      # final_lr = initial_lr/final_div_factor
}
INPUT_STRIDE = SEQUENCE_LENGTH // 2
EPOCHS_N = 30
SAVE_MODEL_EACH_N_ACUMMULATED_BATCHES = 20
LOAD_FROM_EPOCH = None

def get_data_loader_from_file(file_path: pathlib.Path) -> torch.utils.data.DataLoader:
    loaded_np_tokens = np.load(file_path, allow_pickle=False)
    loaded_torch_tokens = torch.tensor(loaded_np_tokens, dtype=torch.long, requires_grad=False).to(DEVICE)

    sequences = []
    max_start_index = len(loaded_np_tokens) - SEQUENCE_LENGTH - 1
    for start_index in range(0, max_start_index, INPUT_STRIDE):
        input_seq = loaded_torch_tokens[start_index:start_index + SEQUENCE_LENGTH]
        output_seq = loaded_torch_tokens[start_index + 1:start_index + SEQUENCE_LENGTH + 1]
        sequences.append((input_seq, output_seq))

    data_loader = torch.utils.data.DataLoader(sequences, batch_size=BATCH_SIZE, shuffle=True)  # type: ignore
    return data_loader

def get_model_save_path(epoch: int) -> pathlib.Path:
    return pathlib.Path(MODEL_SAVE_DIR, f"epoch_{epoch:03d}.pt")

def get_checkpoint_path(epoch: int) -> pathlib.Path:
    return pathlib.Path(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch:03d}.pt")

def save_checkpoint(epoch, optimizer, scheduler, path):
    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Full checkpoint saved to {path}")

def load_checkpoint(optimizer, scheduler, path):
    checkpoint = torch.load(path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch+1}")
    return epoch + 1

def train_routine(resume_from_checkpoint: int | None = None):
    # create save directory if it doesn't exist
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # save gpt config
    with open(pathlib.Path(MODEL_SAVE_DIR, "config.json"), 'w') as f:
        json.dump(GPT_CONFIG, f)

    log_header = "epoch, time, learning_rate, avg_perplexity, avg_perplexity_validation"
    
    # Initialize or append to log file based on whether we're resuming
    log_mode = 'a' if resume_from_checkpoint is not None else 'w'
    with open(pathlib.Path(MODEL_SAVE_DIR, "log.log"), log_mode) as f:
        if log_mode == 'w':
            f.write(log_header + "\n")

    gpt = transformer.MyTransformer(**GPT_CONFIG).to(DEVICE)  # we do not move it to desired float precision - we will do it with autocast
    print("Compiling model")
    gpt: transformer.MyTransformer = torch.compile(gpt)  # type: ignore
    
    # If we're not resuming, save the initial model
    if resume_from_checkpoint is None:
        gpt.save(get_model_save_path(0))

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Using {DEVICE} device")
    print(f"Model has {gpt.parameters_number():_} parameters")
    print("Preparing batches")

    data_loader = get_data_loader_from_file(TOKENIZED_TRAINING_DATA_PATH)
    validation_data_loader = get_data_loader_from_file(TOKENIZED_VALIDATION_DATA_PATH)
    
    total_steps = (len(data_loader) // ACCUMULATION_STEPS) * EPOCHS_N
    print(f"Total training steps with gradient accumulation: {total_steps}")

    scheduler_kwargs = SCHEDULER_KWARGS.copy()
    scheduler_kwargs["total_steps"] = total_steps + 1
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = SCHEDULER(optimizer, **scheduler_kwargs)
    
    # Load optimizer and scheduler state if resuming
    start_epoch = 0
    if resume_from_checkpoint is not None:
        # Load model weights
        gpt.load(get_model_save_path(resume_from_checkpoint))
        # Load optimizer and scheduler state
        start_epoch = load_checkpoint(optimizer, scheduler, get_checkpoint_path(resume_from_checkpoint))
    
    print(f"Training with effective batch size: {VIRTUAL_BATCH_SIZE} (actual: {BATCH_SIZE}, accumulation: {ACCUMULATION_STEPS})")
    print(f"Starting from epoch {start_epoch}")
    print("Start training")

    loss = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()  # Updated to use the correct GradScaler
    for epoch in range(start_epoch, EPOCHS_N):
        time_start = time.time()
        perplexity_sum = 0
        perplexity_sum_accum = 0
        batch_accum_start_time = time.time()
        
        for batch_idx, (input_batch, output_batch) in enumerate(data_loader):
            # Using autocast for mixed precision
            with autocast(device_type=DEVICE, dtype=FLOAT_AUTOCAST_MODE):
                # Forward pass
                predicted = gpt(input_batch)
                predicted_for_loss = predicted.view(-1, predicted.size(-1))
                output_batch_for_loss = output_batch.view(-1)

                # Calculate loss and scale by accumulation steps
                loss_here = loss(predicted_for_loss, output_batch_for_loss) / ACCUMULATION_STEPS
                perplexity_here = math.exp(loss_here.item() * ACCUMULATION_STEPS)
                perplexity_sum += perplexity_here
                perplexity_sum_accum += perplexity_here

            # Backward pass
            scaler.scale(loss_here).backward()
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or batch_idx == len(data_loader) - 1:
                # Clip gradients and update
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                perplexity_avg_accum = perplexity_sum_accum / ACCUMULATION_STEPS
                perplexity_sum_accum = 0
                batch_accum_time_spend = time.time() - batch_accum_start_time
                accum_batch = (batch_idx // ACCUMULATION_STEPS) + 1
                print(f"Epoch {epoch}, Accumulated Batch {accum_batch}/{len(data_loader)//ACCUMULATION_STEPS}: {perplexity_avg_accum:.2f}, LR: {scheduler.get_last_lr()[0]:.8f}, Time: {batch_accum_time_spend:.2f}s", flush=True)
                if accum_batch % SAVE_MODEL_EACH_N_ACUMMULATED_BATCHES == 0:
                    gpt.save(get_model_save_path(epoch))
                    save_checkpoint(epoch, optimizer, scheduler, get_checkpoint_path(epoch))
                batch_accum_start_time = time.time()

        perplexity_sum_validation = 0
        with torch.no_grad():
            for input_batch, output_batch in validation_data_loader:
                with autocast(device_type=DEVICE, dtype=FLOAT_AUTOCAST_MODE):
                    predicted = gpt(input_batch)
                    predicted_for_loss = predicted.view(-1, predicted.size(-1))
                    output_batch_for_loss = output_batch.view(-1)
                    loss_here = loss(predicted_for_loss, output_batch_for_loss)
                    perplexity_here = math.exp(loss_here.item())
                    perplexity_sum_validation += perplexity_here

        avg_perplexity = (perplexity_sum if perplexity_sum < 1e20 else 1e20) / len(data_loader)
        avg_perplexity_validation = (perplexity_sum_validation if perplexity_sum_validation < 1e20 else 1e20) / len(validation_data_loader)
        time_end = time.time()
        
        gpt.save(get_model_save_path(epoch))
        save_checkpoint(epoch, optimizer, scheduler, get_checkpoint_path(epoch))
        
        log_text = f"{epoch}, {time_end - time_start:.2f}, {scheduler.get_last_lr()[0]:.8f}, {avg_perplexity:.4f}, {avg_perplexity_validation:.4f}"
        with open(pathlib.Path(MODEL_SAVE_DIR, "log.log"), 'a') as f:
            f.write(log_text + "\n")
        print(log_header + "\n" + log_text + "\n\n\n")

if __name__ == "__main__":
    train_routine(LOAD_FROM_EPOCH)