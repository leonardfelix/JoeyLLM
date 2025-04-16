# from datasets import load_dataset
# ds = load_dataset("SouthernCrossAI/Project_Gutenberg_Australia", data_dir="data")
# print(ds['train'].features)
from train_gpt2 import PGADataset

# Test the dataset
test_ds = PGADataset(tokenizer)
sample = test_ds[0]
print("Input shape:", sample[0].shape)
print("Sample input:", tokenizer.decode(sample[0][0]))
print("Sample label:", tokenizer.decode(sample[1][0]))