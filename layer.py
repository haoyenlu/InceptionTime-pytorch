import torch
import torch.nn as nn
import math




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


