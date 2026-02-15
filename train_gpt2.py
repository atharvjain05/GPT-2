import torch.nn as nn 
import torch 
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

VOCAB_SIZE = 50257
N_EMBED = 768
N_LAYER = 12 
N_HEAD = 12
CONTEXT_LEN = 1024

class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, n_embed, n_head, context_len):
        super().__init__()
        self.n_head = n_head 
        self.head_size = n_embed // n_head 
        self.c_attn = nn.Linear(n_embed, 3 * n_embed) 
        self.c_proj = nn.Linear(n_embed, n_embed) 
        self.register_buffer("bias", torch.tril(torch.ones(context_len, context_len)).view(1, 1, context_len, context_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embed = x.shape 
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)

        attn_matrix = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        attn_matrix = attn_matrix.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = F.softmax(attn_matrix, dim=-1)
        attn_output = attn_weights @ v 

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, n_embed) 
        return self.c_proj(attn_output) 
        
class TransformerBlock(nn.Module): 
    def __init__(self, n_embed, n_head, context_len): 
        super().__init__()
        self.attn = Attention(n_embed, n_head, context_len)
        self.mlp = MLP(n_embed) 

        self.ln_1 = nn.LayerNorm(n_embed)
        self.ln_2 = nn.LayerNorm(n_embed)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x)) 
        return x
    
class GPT2(nn.Module): 
    
    def __init__(self, vocab_size, n_embed, n_head, n_layer, context_len):
        super().__init__()
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embed),
            wpe = nn.Embedding(context_len, n_embed),
            h = nn.ModuleList([TransformerBlock(n_embed, n_head, context_len) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embed) 
        ))
        
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)


        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape 
        assert seq_len <= CONTEXT_LEN
        embeddings = self.transformer.wte(x) + self.transformer.wpe(torch.arange(seq_len, device=x.device))
        
        for block in self.transformer.h:
            embeddings = block(embeddings) 

        embeddings = self.transformer.ln_f(embeddings)
        logits = self.lm_head(embeddings)
        return logits

    @classmethod
    def from_pretrained(cls):
        model = GPT2(VOCAB_SIZE, N_EMBED, N_HEAD, N_LAYER, CONTEXT_LEN)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()

        # HF GPT-2 uses Conv1D which stores weights transposed vs nn.Linear
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_hf:
            if k.endswith('.attn.bias') or k.endswith('.attn.masked_bias'):
                continue
            with torch.no_grad():
                if any(k.endswith(t) for t in transposed):
                    sd[k].copy_(sd_hf[k].t())
                else:
                    sd[k].copy_(sd_hf[k])

        return model

