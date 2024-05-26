import torch
import torch.nn as nn


class Conv_MLP(nn.Module):
    def __init__(self,in_dim,out_dim,resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_dim,out_dim,3,stride=1,padding=1),
            nn.Dropout(p=resid_pdrop)
        )

    def forward(self,x):
        return self.sequential(x)

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
    self.act = nn.ReLU()

    self.attn = AttentionBlock(
        n_embd = (len(kernels)+ 1) * filter_size,
        n_head=4,
        mlp_hidden_time = 2,
        drop=0.4
    )

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

    if self.use_attn:
      x = x.permute((0,2,1))
      x = self.attn(x)
      x = x.permute((0,2,1))
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

class AttentionBlock(nn.Sequential):
  def __init__(self,n_embd,n_head,mlp_hidden_time,drop = 0.1):
    super(AttentionBlock,self).__init__()
    self.attn =  FullAttention(n_embd,n_head)
    self.ln = nn.LayerNorm(n_embd)
    self.mlp = nn.Sequential(
        nn.Linear(n_embd, 2 * mlp_hidden_time),
        nn.GELU(),
        nn.Linear(mlp_hidden_time * 2, n_embd),
        nn.Dropout(drop),
    )

  def forward(self,x):
    x = x + self.attn(self.ln(x))
    x = x + self.mlp(self.ln(x))
    return x



class InceptionTime(nn.Module):
  def __init__(self,sequence_len,feature_size,label_dim, filter_size=32,use_residual=True, use_bottleneck=True,depth=6,kernels = [10,20,40],use_attn=False):
    super(InceptionTime,self).__init__()
    self.sequence_len = sequence_len
    self.feature_size = feature_size
    self.label_dim = label_dim
    self.depth = depth
    self.use_residual = use_residual
    self.use_attn = use_attn
    self.layers = 4


    self.inceptions = []
    self.shortcuts = []

    prev = feature_size
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