"""
Custom GPT-2 Training Script with Multi-GPU Support

Features:
- Trains on Project Gutenberg Australia dataset
- Automatic mixed precision training
- Multi-GPU support with DataParallel/DistributedDataParallel
- Weights & Biases logging
- Model checkpointing
"""

import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel
from datasets import load_dataset
from transformers import GPT2Tokenizer
from custom_gpt2_model import GPT2Config, GPT2Model  # Your custom implementation

# Configuration
class TrainingConfig:
    def __init__(self):
        self.model_size = "small"            # small/medium/large/xl
        self.batch_size = 2                  # Per GPU batch size
        self.max_seq_len = 1024               # Input sequence length
        self.learning_rate = 3e-5
        self.num_epochs = 1
        self.log_interval = 100              # Steps between logging
        self.save_dir = "checkpoints"
        #self.wandb_project = "GPT2-Training"
        self.use_multi_gpu = torch.cuda.device_count() > 1

# Dataset Class
class PGADataset(Dataset):
    def __init__(self, tokenizer, max_length=1024, use_pre_tokenized=True):
        self.dataset = load_dataset(
            "SouthernCrossAI/Project_Gutenberg_Australia",
            data_dir="data",
            split="train"
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_pre_tokenized = use_pre_tokenized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.use_pre_tokenized:
            tokens = torch.tensor(self.dataset[idx]["cl100k_base"][:self.max_length])
            return tokens[:-1], tokens[1:]
        else:
            text = self.dataset[idx]["Paragraph Text"]
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True
            ).squeeze(0)  # Remove batch dimension
            return tokens[:-1], tokens[1:]
        
def collate_fn(batch):
    inputs, labels = zip(*batch)
    
    # Find max length in batch
    max_len = max(len(x) for x in inputs)
    
    # Pad sequences
    padded_inputs = []
    padded_labels = []
    
    for input_seq, label_seq in zip(inputs, labels):
        pad_amount = max_len - len(input_seq)
        padded_inputs.append(torch.cat([input_seq, torch.zeros(pad_amount, dtype=torch.long)]))
        padded_labels.append(torch.cat([label_seq, torch.full((pad_amount,), -100, dtype=torch.long)]))
    
    return torch.stack(padded_inputs), torch.stack(padded_labels)
        
# Tiny Model Configuration
def get_tiny_config():
    return GPT2Config(
        vocab_size=50257,
        max_seq_len=512,
        hidden_dim=512,      # Reduced from 768
        num_heads=8,         # Reduced from 12
        num_layers=2,        # Only 2 decoder blocks
        dropout=0.1
    )

# Initialize Training
config = TrainingConfig()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Model Setup
model_config = get_tiny_config()
model = GPT2Model(model_config)

# Load custom weights or initialize from HF
# Uncomment if you want to start from pretrained weights
# from load_hf_weights import load_hf_weights
# model = load_hf_weights(model, "gpt2")

# Multi-GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config.use_multi_gpu:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)
model.to(device)

# Optimizer & Loss
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()

# W&B Initialization
# wandb.init(project=config.wandb_project, config=vars(config))
# wandb.watch(model)

# Training Loop
def train():
    # dataset = PGADataset(tokenizer, config.max_seq_len)

    # Initialize dataset with pre-tokenized data
    dataset = PGADataset(tokenizer=tokenizer, max_length=config.max_seq_len, use_pre_tokenized=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        for step, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, model_config.vocab_size), 
                            labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Logging
            if step % config.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                wandb.log({
                    "loss": avg_loss,
                    "lr": optimizer.param_groups[0]['lr']
                })
                
                print(f"Epoch {epoch+1} | Step {step} | Loss: {avg_loss:.4f}")

                # Generate sample text
                sample_input = "Once upon a time"
                generate_sample(sample_input, model, tokenizer, device)

        # Save checkpoint
        save_path = f"{config.save_dir}/epoch_{epoch+1}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, save_path)
        print(f"💾 Saved checkpoint to {save_path}")

def generate_sample(prompt, model, tokenizer, device, max_length=50):
    """Generate text sample for monitoring progress"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token = outputs.argmax(-1)[:, -1]
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
            if next_token == tokenizer.eos_token_id:
                break
    
    generated = tokenizer.decode(input_ids[0])
    wandb.log({"sample": wandb.Html(f"<pre>{generated}</pre>")})

if __name__ == "__main__":
    train()
    print("✅ Training complete!")