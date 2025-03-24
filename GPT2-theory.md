📘 GPT-2 Overview: Autoregressive Model with Masked Self-Attention
🧠 What Is an Autoregressive Model?
An autoregressive (AR) model predicts each token based on all the previous ones. In GPT-2:

P(x₁, x₂, ..., xₙ) = P(x₁) · P(x₂ | x₁) · P(x₃ | x₁, x₂) ··· P(xₙ | x₁, ..., xₙ₋₁)

This means GPT-2 generates text one token at a time, from left to right, and never looks ahead.

🤖 GPT-2 Architecture
GPT-2 is a generative model based on the Transformer architecture

It only uses the Decoder blocks (no Encoder)

The Decoder is repeated N times (e.g., 12 blocks for GPT-2 small)

GPT-2 is fully autoregressive, optimized for text generation

🔄 GPT-2 Tokenization and Embedding Pipeline
1. Tokenization
Input text is broken into smaller chunks called tokens using Byte Pair Encoding (BPE). Each token maps to a unique Token ID from the vocabulary V.

Example:
"Hello world!" → ["Hello", " world", "!"] → [15496, 995, 0]

2. Embedding Layer
Each Token ID is converted into a 768-dimensional vector using a learned embedding matrix E ∈ ℝ^{|V| × 768}.

Embedding(Token ID) = E[Token ID]

This gives us:

A continuous representation of discrete tokens

Input shape: seq_len × d_model

🧠 Self-Attention in GPT-2
What is Self-Attention?
Self-Attention helps the model determine which tokens should pay attention to which others. It computes:

Attention(Q, K, V) = Softmax(QKᵀ / √dₖ) · V

Each input token is projected into:

Q: Query

K: Key

V: Value

These are obtained via linear transformations:

Q = X · W_Q
K = X · W_K
V = X · W_V

For single-head attention:

W_Q, W_K, W_V ∈ ℝ^{768 × 64}

For multi-head attention (12 heads):

W_Q, W_K, W_V ∈ ℝ^{768 × 768}

🧩 Multi-Head vs Single-Head Attention
In Multi-Head Attention, multiple attention "heads" operate in parallel:

Each head focuses on different types of relationships (e.g., syntactic, semantic)

The outputs of all heads are concatenated:

Concat(head₁, ..., head₁₂) ∈ ℝ^{seq_len × 768}

Then projected back to original space using W_O ∈ ℝ^{768 × 768}

Final Output = Multi-Head Output · W_O

Note: W_O is a trainable parameter learned during model training.

🔒 Masked Multi-Head Self-Attention in GPT-2
In GPT-2, we use Masked Self-Attention to ensure causality:

Each token can only attend to previous tokens

Future tokens are masked out using a triangular mask:

M_{i,j} =
  0   if j ≤ i
  -∞  if j > i

Applied before Softmax to prevent "seeing the future"

Why?
👉 To ensure GPT-2 predicts token x_t using only x₁, ..., xₜ₋₁

⚙️ Feed-Forward Network (FFN)
After attention, GPT-2 passes the output through a two-layer position-wise feed-forward network:

FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂

Applies to each token independently

Allows complex feature transformation

Hidden layer size is often 4× the model dimension (e.g., 3072 if d_model = 768)

🔁 GPT-2 Text Generation Loop
To generate a sentence:

Input: "The weather"

Predict: "is"

New input: "The weather is"

Predict: "nice"

Repeat until stop condition

Each step:

Embeds current input

Computes Masked Self-Attention

Uses output to predict next token

This process is autoregressive, generating one token at a time.

✅ Summary
Component	Description
Autoregressive	Predicts one token at a time, based on past tokens
Decoder-only Model	No encoder, only decoder blocks
Masked Self-Attention	Prevents attention to future tokens
Multi-Head Attention	Multiple attention heads, merged and projected
Feed-Forward Network	Applies deep transformation per token
Embedding + Tokenization	Converts input text to numerical vectors
