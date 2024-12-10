import tiktoken
import torch

class DataLoader:
    """
    A really basic iterator over some input data.
    """

    def __init__(self, encoding, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            data = f.read()

        enc = tiktoken.get_encoding(encoding)
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"One epoch amounts to {len(self.tokens) // (B * T)} minibatches")

        self.current_position = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        # If current position is out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            raise StopIteration
        
        return x, y