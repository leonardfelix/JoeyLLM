import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from custom_gpt2_model import GPT2Config, GPT2Model

# Configuration
class TrainingConfig:
    def __init__(self):
        self.model_size = "tiny"
        self.batch_size = 2
        self.max_seq_len = 512  # Must match model config
        self.learning_rate = 3e-5
        self.num_epochs = 1
        self.log_interval = 100

# Dataset Class
class PGADataset(Dataset):
    def __init__(self, tokenizer, max_length=512):
        self.dataset = load_dataset(
            "SouthernCrossAI/Project_Gutenberg_Australia",
            data_dir="data",
            split="train"
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["Paragraph Text"]
        
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        ).squeeze(0)
        
        return tokens[:-1], tokens[1:]

# Collate Function
def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_len = max(len(x) for x in inputs)
    
    padded_inputs = torch.stack([
        torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) 
        for x in inputs
    ])
    
    padded_labels = torch.stack([
        torch.cat([y, torch.full((max_len - len(y),), -100, dtype=torch.long)])
        for y in labels
    ])
    
    return padded_inputs, padded_labels

# Tiny Model Config
def get_tiny_config():
    return GPT2Config(
        vocab_size=50257,
        max_seq_len=512,
        hidden_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    )

# Training Loop
def train():
    config = TrainingConfig()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    tiny_config = GPT2Config(model_size="tiny")  # Uses predefined tiny config
    model = GPT2Model(tiny_config).to("cuda")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    dataset = PGADataset(tokenizer, model_config.max_seq_len)
    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    for epoch in range(config.num_epochs):
        model.train()
        for step, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model_config.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            if step % config.log_interval == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
    print("✅ Training complete!")