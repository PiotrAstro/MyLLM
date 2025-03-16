import os
import datasets
import sys

# Limit to certain size

DATASET_NAME = "openwebtext"
PATH = os.path.join("data", "dataopenwebtext")
TARGET_TRAIN_SIZE_BYTES = 100 * 1024 * 1024  # 500MB
TARGET_TEST_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
SIZE_PER_FILE = 10 * 1024 * 1024  # 10MB
DOCUMENT_SEPARATOR = "<|endoftext|>"

def process(dataset_iterator, max_size, single_file_size, path, separator):
    def safe_file(doc_num):
        with open(os.path.join(path, f"{doc_num:03d}.txt"), "w", encoding="utf-8") as f:
                for i in range(len(file_texts)):
                    text = file_texts[i]
                    f.write(text)
                    if i != len(file_texts) - 1:
                        f.write(separator)

    documents = []
    current_size = 0
    for i, item in enumerate(dataset_iterator):
        text_size = sys.getsizeof(item["text"])
        if current_size + text_size <= max_size:
            documents.append(item["text"])
            current_size += text_size
        else:
            break
        
        # Print progress
        if i % 1000 == 0:
            print(f"Processed {i} documents, current size: {current_size / (1024*1024):.2f} MB")
    
    print(f"Collected {len(documents)} documents, total size: {current_size / (1024*1024):.2f} MB")
    
    # split it to several files
    current_document_num = 0
    file_texts = []
    current_size = 0
    for doc in documents:
        current_size += sys.getsizeof(doc)
        file_texts.append(doc)
        if current_size >= single_file_size:
            safe_file(current_document_num)
            current_document_num += 1
            file_texts = []
            current_size = 0
    if file_texts:
        safe_file(current_document_num)


if __name__ == "__main__":
    dataset = datasets.load_dataset(DATASET_NAME, streaming=True, trust_remote_code=True)
    os.mkdir(PATH)
    train_path = os.path.join(PATH, "train")
    test_path = os.path.join(PATH, "validation")
    os.mkdir(train_path)
    os.mkdir(test_path)

    iterator = dataset["train"].__iter__()
    process(iterator, TARGET_TRAIN_SIZE_BYTES, SIZE_PER_FILE, train_path, DOCUMENT_SEPARATOR)
    # dataset i have tested doesnt have test set, so I use same iterator to get some next train data and use it as test data
    process(iterator, TARGET_TEST_SIZE_BYTES, SIZE_PER_FILE, test_path, DOCUMENT_SEPARATOR)

    print("Dataset downloaded")
