import os
from datasets import Dataset
from transformers import BertTokenizer, BertForMaskedLM, TrainingArguments, Trainer
import Corpus
import torch


def mask_entities(text, entities, mask_token='[MASK]'):
    for entity in entities:
        text = ' '.join(text)  # Delimiter is space
        text = text.replace(entity, mask_token)
        text = text.split()
    return text

def split_text(text, max_len, tokenizer, mask_token='[MASK]'):
    words = text.split()
    current_chunk = []
    chunks = []

    for word in words:
        # Simulate what would happen if we add this word to the current chunk
        trial_chunk = current_chunk + [word]
        trial_text = ' '.join(trial_chunk)
        trial_tokens = tokenizer.tokenize(trial_text)

        # Check if adding this word would exceed the maximum length
        if len(trial_tokens) > max_len:
            # Start a new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            # Otherwise, add the word to the current chunk
            current_chunk = trial_chunk

    # Add the last chunk if any remains
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def load_and_prepare_data(books_objects, entities, tokenizer):
    all_texts = []
    max_len = 510  # BERT's max sequence length - 2 for [CLS] and [SEP]
    for book in books_objects:
        line = book.one_ref_list
        masked_line = mask_entities(line, entities)
        # Split the text to fit into BERT's input size
        text_chunks = split_text(masked_line, max_len, tokenizer)
        all_texts.extend(text_chunks)
    return all_texts

def load_entities(file_path):
    entities = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                names = line.strip().split()
                entities.extend(names)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return entities

# Tokenizer initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load entities and prepare text data
entities = load_entities("../Data/Harry_Potter_Books/Entities")
book_object_list = Corpus.Corpus_Main('../Data/Harry_Potter_Books', "../test/Harry Potter Corpus", 7, 200, 7, 15, "Harry_Potter")
texts = load_and_prepare_data(book_object_list, entities, tokenizer)
data = Dataset.from_dict({"text": texts})

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenized_inputs.input_ids.clone()
    labels[labels != tokenizer.mask_token_id] = -100  # Only calculate loss for masked tokens
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = data.map(tokenize_and_align_labels, batched=True)

# Load and Fine-Tune the Model
model = BertForMaskedLM.from_pretrained('bert-base-uncased').to("cuda" if torch.cuda.is_available() else "cpu")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets
)

trainer.train()

# Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_bert")
