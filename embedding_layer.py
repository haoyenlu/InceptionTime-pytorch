import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module): # NLC
  def __init__(self,d_model,max_len=1024):
    super(PositionalEmbedding,self).__init__()
    pe = torch.zeros(max_len,d_model).float()
    pe.require_grad = False

    position = torch.arange(0,max_len).float().unsqueeze(1)
    div_term = (torch.arange(0,d_model,2).float() * -(math.log(10000.0) / d_model)).exp()

    pe[:,0::2] = torch.sin(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe',pe)

  def forward(self,x):
    return self.pe[:,:x.size(1)] + x
  
  
class LearnablePositionalEncoding(nn.Module): #NLC
    def __init__(self,d_model,dropout=0.1,max_len=1024):
        super(LearnablePositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1,max_len,d_model))
        nn.init.uniform_(self.pe,-0.02,0.02)

    def forward(self,x):
        x = x + self.pe
        return self.dropout(x)


class TokenEmbedding(nn.Module): # NLC
  def __init__(self,c_in,d_model):
    super(TokenEmbedding,self).__init__()
    padding = 1 if torch.__version__ >= '1.5.0' else 2
    self.tokenConv = nn.Conv1d(in_channels=c_in,out_channels=d_model,
                               kernel_size=1,padding='same',bias=False)
    for m in self.modules():
      if isinstance(m,nn.Conv1d):
        nn.init.kaiming_normal(m.weight,mode='fan_in',nonlinearity='leaky_relu')

  def forward(self,x):
    x = self.tokenConv(x.permute(0,2,1)).transpose(1,2)
    return x


class DataEmbedding(nn.Module):
  def __init__(self,c_in,d_model,dropout=0.1,max_len=1024):
    super(DataEmbedding,self).__init__()
    self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
    self.position_embedding = PositionalEmbedding(d_model=d_model,max_len=max_len)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    x = self.value_embedding(x) + self.position_embedding(x)
    return self.dropout(x)