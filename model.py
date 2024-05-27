import torch
import torch.nn as nn

from layer import InceptionModule, ResidualLayer, DataEmbedding


class InceptionTime(nn.Module):
  def __init__(self,sequence_len,feature_size,label_dim, 
               filter_size=32,depth=6,kernels = [10,20,40],dropout=0.2,
               use_residual=True, use_bottleneck=True,use_attn=False):
    
    super(InceptionTime,self).__init__()
    self.sequence_len = sequence_len
    self.feature_size = feature_size
    self.label_dim = label_dim
    self.depth = depth
    self.use_residual = use_residual
    self.use_attn = use_attn
    
    self.embedding = DataEmbedding(c_in = feature_size,d_model=filter_size,dropout=dropout,max_len=sequence_len)

    self.inceptions = []
    self.shortcuts = []

    prev = filter_size
    residual_prev = prev

    for d in range(depth):
      self.inceptions.append(InceptionModule(
          prev,
          filter_size,
          kernels,
          use_bottleneck,
          use_attn = use_attn,
      ))

      if use_residual and d % 3 == 2: # 2,5
        self.shortcuts.append(ResidualLayer(
            input_dim = residual_prev,
            output_dim = (len(kernels)+1) * filter_size
        ))
        residual_prev = prev

      prev = (len(kernels) + 1) * filter_size

    self.inceptions = nn.ModuleList(self.inceptions)
    self.shortcuts = nn.ModuleList(self.shortcuts)
    self.out = nn.Sequential(
      nn.Linear(prev,label_dim * 2),
      nn.ReLU(),
      nn.Linear(label_dim * 2,label_dim),
      nn.Softmax()
    )

  def forward(self,x): # input shape: (N,C,L)
    assert self.sequence_len == x.shape[2] and self.feature_size == x.shape[1]

    x = self.embedding(x.permute((0,2,1))).permute((0,2,1))

    res_input = x
    s_index = 0
    for d in range(self.depth):
      x = self.inceptions[d](x)

      if self.use_residual and d % 3 == 2:
        x = self.shortcuts[s_index](res_input,x)
        res_input = x
        s_index += 1

    x = torch.mean(x,dim=2)
    x = self.out(x)

    return x