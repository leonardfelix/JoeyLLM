from datasets import load_dataset
import torch
from transformers import AutoTokenizer

# Load the dataset from Hugging Face
dataset = load_dataset("SouthernCrossAI/Project_Gutenberg_Australia")

# Load GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Function to tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Save tokenized dataset
tokenized_dataset.save_to_disk("tokenized_data")
print("Tokenized dataset saved successfully!")
