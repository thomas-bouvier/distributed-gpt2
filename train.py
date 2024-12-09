import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    """
    Multiple heads are managed in this class.
    """

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_heads == 0 # heads must add up to the number of embeddings

        self.n_heads = config.n_heads

        # Key, query, and value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

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
        k = k.view(B, T, self.n_heads, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

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
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
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

    def forward(self):
        pass

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load pre-trained GPT-2 model weights from Hugging Face.

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
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

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


model = GPT.from_pretrained("gpt2")
