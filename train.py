import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.nn import functional as F

from load import DataLoader

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    """Multiple heads are managed in this class."""

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_heads == 0 # heads must add up to the number of embeddings

        self.n_heads = config.n_heads
        self.n_embd = config.n_embd

        # Key, query, and value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # flag to scale residual weights at init

        # Not really a bias, more of a mask. But following OpenAI naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape # Batch, Sequence length, Embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # Attention, materialize the large (T, T) matrix for all queries and keys
        att = (q @ k.transpose(-2, -1)) * k.shape[-1]**-0.5 # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # (B, nh, T, T)
        att = F.softmax(att, dim=-1) # make logits sum to one, (B, nh, T, T)

        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # flag to scale residual weights at init
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) # residual layer #1
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # residual layer #2
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # communication, reduce operation
        x = x + self.mlp(self.ln_2(x)) # computation, map operation
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Reusing the same notation as HF's transformers library
        self.transformer = nn.ModuleDict(dict(
            # An Embedding is just a glorified tensor
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # embeddings to logits

        # Weight sharing scheme between token embeddings and output linear layers.
        # This saves weights, as suggested in the Attention paper
        self.transformer.wte.weight = self.lm_head.weight

        # Init params by iterating on sub-modules
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02

        if isinstance(module, nn.Linear):
            # From the GPT-2 paper, weights of residual layers should be scaled
            # by N ** -0.5 at init, N being the number of residual layers.
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5 # x2 because there are two residual layers in each Block

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        x = tok_emb + pos_emb

        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pre-trained GPT-2 model weights from Hugging Face.

        Attributes defined in our GPT must match with the naming of original
        weights from OpenAI.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        print(f"Loading weights from pretrained GPT: {model_type}")
        from transformers import GPT2LMHeadModel

        # n_layers, n_heads and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # Create a from-scratch initialized minGPT model
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        # Ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        # Same, just the mask (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # OpenAI checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # This means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


torch.manual_seed(1337)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")

num_return_sequences=5
max_length = 50
model_name = "gpt2"
epochs = 50
lr = 3e-4
bs = 4

# Load some input data
train_loader = DataLoader(encoding=model_name, B=4, T=bs)

model = GPT(GPTConfig())
model.eval()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, loss: {loss.item()}")

import sys; sys.exit(0)

####################################
# Inference
####################################

def prefix_tokens():
    import tiktoken

    enc = tiktoken.get_encoding(model_name)
    tokens = enc.encode("Hello world,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (3,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num_return_sequences, 3)
    return tokens.to(device)

model = GPT.from_pretrained(model_name)
model.eval()
model.to(device)

x = prefix_tokens()

with torch.no_grad():
    while x.shape[1] < max_length:
        logits, _ = model(x) # (B, T, vocab_size)

        # Take the last logits
        logits = logits[:, -1, :]

        # Get the probabilities
        probs = F.softmax(logits, dim=-1)

        # Top-k sampling of 50 (HF's default)
        # We don't want to sample very rare tokens.
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)

        # Select a single token from the top-k probs
        ix = torch.multinomial(topk_probs, 1) # (B, 1)

        # Gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

        # Append to the sequence
        x = torch.cat((x, xcol), dim=1)

# Print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"> {decoded}")
