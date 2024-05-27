import torch
import torch.nn as nn
import math

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = torch.nn.functional.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class EncoderBlock(nn.Module):
    def __init__(self,d_model,d_hidden,n_head,dropout):
        super(EncoderBlock,self).__init__()
        self.attn = FullAttention(n_embd=d_model,n_head=n_head,attn_pdrop=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fn = nn.Sequential(
            nn.Linear(d_model,d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden,d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):
       _x = x
       x = self.attn(x)
       x = self.dropout1(x)
       x = self.ln1(x + _x)

       _x = x
       x = self.fn(x)
       x = self.dropout2(x)
       x = self.ln2(x + _x)
       return x
   
class Encoder(nn.Module):
    def __init__(self,n_layers,d_model,fn_hidden,n_head,dropout):
        super(Encoder,self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model,
                         d_hidden=fn_hidden,
                         n_head=n_head,
                         dropout=dropout)
        for _ in range(n_layers)])
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x