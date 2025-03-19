import torch
import argparse
from custom_gpt2_model import GPT2Config, GPT2Model, load_hf_weights
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize argument parser
parser = argparse.ArgumentParser(description='Generate text with custom GPT-2')
parser.add_argument('--model_size', type=str, default='small',
                    choices=['small', 'medium', 'large', 'xl'],
                    help='Model size (small, medium, large, xl)')
parser.add_argument('--max_length', type=int, default=100,
                    help='Maximum number of tokens to generate')
parser.add_argument('--temperature', type=float, default=0.9,
                    help='Temperature for sampling (0.1-1.0)')
parser.add_argument('--top_p', type=float, default=0.92,
                    help='Top-p (nucleus) sampling threshold (0.5-1.0)')
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Model configuration based on size
model_configs = {
    "small": {"hidden_dim": 768, "num_heads": 12, "num_layers": 12},
    "medium": {"hidden_dim": 1024, "num_heads": 16, "num_layers": 24},
    "large": {"hidden_dim": 1280, "num_heads": 20, "num_layers": 36},
    "xl": {"hidden_dim": 1600, "num_heads": 25, "num_layers": 48}
}

# Initialize model
config = GPT2Config(
    hidden_dim=model_configs[args.model_size]["hidden_dim"],
    num_heads=model_configs[args.model_size]["num_heads"],
    num_layers=model_configs[args.model_size]["num_layers"]
)
model = GPT2Model(config)

# Load weights from Hugging Face
hf_model_name = {
    "small": "gpt2",
    "medium": "gpt2-medium",
    "large": "gpt2-large",
    "xl": "gpt2-xl"
}[args.model_size]

try:
    # Try loading custom trained weights first
    model.load_state_dict(torch.load(f"hf_{args.model_size}_gpt2.pth"))
    print(f"✅ Loaded custom {args.model_size} model weights")
except FileNotFoundError:
    # Fallback to Hugging Face weights
    print(f"⚠️  Custom weights not found, loading from Hugging Face {hf_model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    model = load_hf_weights(model, hf_model)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# def generate_text(prompt, max_length=50, temperature=0.7):
#     """
#     Improved text generation with temperature sampling
#     """
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
#     # Limit input to 10 tokens
#     if input_ids.shape[1] > 10:
#         input_ids = input_ids[:, :10]
#         print("\n[WARNING] Input truncated to 10 tokens")

#     # Generate text
#     with torch.no_grad():
#         for _ in range(max_length):
#             outputs = model(input_ids)
#             logits = outputs[:, -1, :]
            
#             # Apply temperature
#             logits = logits / temperature
#             probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
#             # Sample from distribution
#             next_token = torch.multinomial(probabilities, 1)
#             input_ids = torch.cat([input_ids, next_token], dim=1)

#             if next_token == tokenizer.eos_token_id:
#                 break

#     return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def generate_text(prompt, max_length=50, temperature=0.9, top_p=0.92):
    """
    Generates text using nucleus sampling with temperature
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Truncate input if needed
    if input_ids.shape[1] > 10:
        input_ids = input_ids[:, :10]
        print("\n[WARNING] Input truncated to 10 tokens")

    # Generate tokens
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Nucleus (top-p) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter indices and remove
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            probs[indices_to_remove] = 0
            
            # Sample from filtered distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if EOS generated
            if next_token == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Get user input
user_prompt = input("\nEnter a prompt (max 10 tokens): ")
generated_text = generate_text(
    user_prompt,
    temperature=0.9,  # More randomness
    top_p=0.92,       # Broader selection
    max_length=100
)
print("\nGenerated Text:\n", generated_text)