import torch
import torch.nn as nn
import math

from embedding_layer import PositionalEmbedding
from transformer_layer import Encoder


class InceptionModule(nn.Module):
  def __init__(self,input_dim,filter_size=32,kernels=[10,20,40],use_bottleneck=True,use_attn=False):
    super(InceptionModule,self).__init__()
    self.bottleneck_size = 32
    self.use_bottleneck = use_bottleneck
    self.use_attn = use_attn
    self.filter_size = filter_size
    self.input_inception = nn.Conv1d(input_dim,self.bottleneck_size,kernel_size=1,padding='same',bias=False)

    self.conv_list = []
    prev = input_dim if not use_bottleneck else self.bottleneck_size
    for kernel in kernels:
      self.conv_list.append(nn.Conv1d(prev,filter_size,kernel_size=kernel,padding='same',bias=False))

    self.conv_list = nn.ModuleList(self.conv_list)

    self.max_pool_1 = nn.MaxPool1d(kernel_size=3,padding=1,stride=1)
    self.conv6 = nn.Conv1d(input_dim,self.filter_size,kernel_size=1,padding='same',bias=False)

    self.bn = nn.BatchNorm1d((len(kernels) + 1) * filter_size)
    self.act = nn.GELU()


  def forward(self,x):
    _x = x
    if self.use_bottleneck:
      x = self.input_inception(x)

    x_list = []
    for conv in self.conv_list:
      x_list.append(conv(x))

    _x = self.max_pool_1(_x)
    x_list.append(self.conv6(_x))

    x = torch.cat(x_list,dim=1)
    x = self.bn(x)
    x = self.act(x)

    return x

class ResidualLayer(nn.Sequential):
  def __init__(self,input_dim,output_dim):
    super(ResidualLayer,self).__init__()
    self.conv = nn.Conv1d(input_dim,output_dim,kernel_size=1,padding='same',bias=False)
    self.bn = nn.BatchNorm1d(output_dim)
    self.act = nn.ReLU()

  def forward(self,residual_input,input):
    residual = self.conv(residual_input)
    residual = self.bn(residual) 

    x = residual + input
    x = self.act(x)
    return x


class Transformer(nn.Module):
  def __init__(self,seq_len,feature_size,label_dim,d_model,n_head,fn_hidden,n_layers,dropout):
    super(Transformer,self).__init__()
    self.encoder_input_layer = nn.Linear(feature_size,d_model)
    self.emb = PositionalEmbedding(d_model,seq_len)
    self.encoder = Encoder(
        n_layers=n_layers,
        d_model=d_model,
        fn_hidden=fn_hidden,
        n_head=n_head,
        dropout=dropout
    )
    self.out = nn.Sequential(
      nn.LayerNorm(d_model),
      nn.Flatten(),
      nn.Linear(d_model * seq_len,512),
      nn.ReLU(),
      nn.Linear(512,256),
      nn.ReLU(),
      nn.Linear(256,128),
      nn.ReLU(),
      nn.Linear(128,label_dim),
      nn.Softmax()
    )

    def forward(self,x):
      x = self.encoder_input_layer(x)
      x = self.emb(x)
      x = self.encoder(x)
      x = self.out(x)
      return x