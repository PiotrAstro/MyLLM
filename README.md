# MyLLM

A self-implemented GPT-2 style language model built from scratch. This project was created with the help of:
- The original paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- OpenAI's ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Reference from [Andrew Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- Insights from [OpenAI's GPT-2 repository](https://github.com/openai/gpt-2)

I like implementing everything from scratch, so this project implements a complete language model stack including:
- Custom BPE tokenizer (following the GPT-2 paper methodology)
- Transformer architecture composed from basic PyTorch building blocks
- Training pipeline with gradient accumulation and mixed precision
- Interactive text generation interface

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/PiotrAstro/MyLLM.git
cd MyLLM

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Project Configuration

### Data Processing Configuration

The data processing pipeline uses configuration in the respective files:

#### `src/data_download.py`
```python
DATASET_NAME = {"path": "HuggingFaceFW/fineweb", "name": "sample-100BT"}  # Dataset to download
PATH = os.path.join("data", "fineweb")  # Where data will be saved
TARGET_TRAIN_SIZE_BYTES = 500 * 1024 * 1024  # Target size for training data (500MB)
TARGET_TEST_SIZE_BYTES = 30 * 1024 * 1024  # Target size for validation data (30MB)
SIZE_PER_FILE = 10 * 1024 * 1024  # Size per individual file (10MB)
DOCUMENT_SEPARATOR = "<|endoftext|>"  # Token to separate documents
```

#### `src/construct_tokenizer.py`
```python
TOKENS_N = 32_768  # Vocabulary size, one token will be <|endoftext|>
DATA_PATH = pathlib.Path("data", "fineweb", "train")  # Path to training data
SAFE_PATH = pathlib.Path("results", f"tokenizer_fineweb_{TOKENS_N}.txt")  # Where to save tokenizer
```

#### `src/tokenize_data.py`
```python
DATA_PATH = pathlib.Path("data", "fineweb", "train")  # Path to raw data
TOKENIZED_SAFE_PATH = pathlib.Path("data", "tokenized_fineweb_32768.npy")  # Path for tokenized data
TOKENIZER_PATH = pathlib.Path("results", "tokenizer_fineweb_32768.txt")  # Path to trained tokenizer
```

### Model Training Configuration

The model training is configured in `src/train_model.py`:

```python
# Data paths
TOKENIZED_TRAINING_DATA_PATH = pathlib.Path("data", "tokenized_fineweb_32768.npy")
TOKENIZED_VALIDATION_DATA_PATH = pathlib.Path("data", "tokenized_fineweb_32768_validation.npy")
TRAINED_TOKENIZER_PATH = pathlib.Path("results", "tokenizer_fineweb_32768.txt")
MODEL_SAVE_DIR = pathlib.Path("results", "model_fineweb_32768_tokens")

# Model architecture
SEQUENCE_LENGTH = 512  # Context window size
GPT_CONFIG = {
    "max_sequence_length": SEQUENCE_LENGTH,
    "embeding_size": 512,  # Embedding dimension
    "feed_forward_hidden_n": 512 * 4,  # Hidden layer size in feed-forward network
    "tokens_number": TOKENIZER.get_tokens_number(),  # should be 32768, will take care of <|endoftext|>
    "num_heads": 8,  # Number of attention heads, "embeding_size" % "num_heads" should be 0
    "blocks_n": 8,  # Number of transformer blocks
    "residual_dropout": 0.05,
    "attention_dropout": 0.05,
    "embeding_dropout": 0.05
}

# Training settings
FLOAT_AUTOCAST_MODE = torch.bfloat16  # Precision for mixed precision training, check you gpu capabilities
BATCH_SIZE = 64  # Batch size per gradient update, experiment with gpu memory
VIRTUAL_BATCH_SIZE = 512  # Effective batch size with gradient accumulation
ACCUMULATION_STEPS = VIRTUAL_BATCH_SIZE // BATCH_SIZE
WEIGHT_DECAY = 0.005  # L2 regularization
LEARNING_RATE = 1e-3  # Base learning rate
SCHEDULER = torch.optim.lr_scheduler.OneCycleLR  # Learning rate scheduler
SCHEDULER_KWARGS = {
    "max_lr": 1e-3,             # Peak learning rate
    "pct_start": 0.1,            # Percentage of training spent in the increasing phase
    "div_factor": 25.0,          # initial_lr = max_lr/div_factor
    "final_div_factor": 100.0    # final_lr = initial_lr/final_div_factor
}
INPUT_STRIDE = SEQUENCE_LENGTH // 2  # Stride for creating training sequences
EPOCHS_N = 30  # Number of training epochs
SAVE_MODEL_EACH_N_ACUMMULATED_BATCHES = 20  # Save checkpoint frequency
LOAD_FROM_EPOCH = None  # Set to an integer to resume from a checkpoint
```

### Running Inference

The interactive text generation is the main user-facing component and can be run from the command line:

```bash
# Run the interactive inference script
python src/live_inference.py
```

Key inference settings in `src/live_inference.py`:
```python
TRAINED_TOKENIZER_PATH = pathlib.Path("results", "tokenizer_fineweb_32768.txt")
MODEL_SAVE_PATH = pathlib.Path("results", "model_32768_tokens", "epoch_029.pt")
CONFIG_SAVE_PATH = pathlib.Path("results", "model_32768_tokens", "config.json")
TEMPERATURE = 0.7  # Higher = more creative, lower = more deterministic
ONE_PROMPT_MAX_OUTPUT_TOKENS = 512  # Maximum tokens to generate per prompt
STREAMING = True  # Whether to stream tokens as they're generated
```

Example session (it was trained on fineweb with on lot of stories, so it works best at continuing them):
```
Enter your text or write 'exit' to quit:
>>>  In the small town of Millfield, where the fog rolled in every evening like clockwork, Sarah discovered an antique pocket watch buried beneath the floorboards of her newly inherited house. The watch was unlike anything she had seen before—its face adorned with strange symbols instead of numbers, and its hands moving counterclockwise at an uneven pace.


PioterLLM >>>  In the late evening, the watch was filled with the secret fire that was set off by a few other people from local schools.
‘You don’t have to play with me,’ said Sarah.
‘You don’t have to do anything. You can’t fight in a frenzy,” said Sarah.
‘This is a great way to think about what you were up to and what you should do to get out of the city.’
```

## Workflow

The typical workflow for this project is:

1. **Data Processing Pipeline**:
   - Download the dataset with `data_download.py`
   - Construct a tokenizer with `construct_tokenizer.py`
   - Tokenize the dataset with `tokenize_data.py`, you should tokenize train set and validation set separately

2. **Model Training**:
   - Configure model architecture in `train_model.py`
   - Run training, which automatically saves checkpoints
   - Monitor training logs in the `MODEL_SAVE_DIR/log.log` file and in the command line

3. **Inference**:
   - Configure the model path in `live_inference.py`
   - Run the interactive interface to generate text

## Project Structure

- `src/transformer/` - Core transformer implementation from scratch
- `src/tokenization/` - BPE tokenizer implementation from scratch
- `src/data_download.py` - Dataset acquisition
- `src/construct_tokenizer.py` - Tokenizer training
- `src/tokenize_data.py` - Dataset preprocessing
- `src/train_model.py` - Model training pipeline
- `src/live_inference.py` - Interactive text generation
- `requirements.txt` - Project dependencies

## Future Work

- Attention visualization tools for model interpretability
- Fine-tuning capabilities for specialized tasks

## License

This project is available under the MIT License.

## Acknowledgements

Special thanks to the authors of all the referenced papers and repositories that made this implementation possible.
