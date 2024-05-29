import torch
import torch.nn as nn

from layer import InceptionModule, ResidualLayer, FCNLayer

from embedding_layer import DataEmbedding

from embedding_layer import PositionalEmbedding
from transformer_layer import Encoder


class InceptionTime(nn.Module):
    def __init__(self,batch_size,sequence_len,feature_size,label_dim, 
                inception_filter=32,fcn_filter = 128,depth=6,fcn_layers = 6,kernels = [10,20,40],dropout=0.2,
                use_residual=True, use_bottleneck=True,use_attn=False,use_embedding=False):
        
        super(InceptionTime,self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sequence_len = sequence_len
        self.feature_size = feature_size
        self.label_dim = label_dim
        self.depth = depth
        self.use_residual = use_residual
        self.use_embedding = use_embedding
        self.filter_size = inception_filter
        self.batch_size = batch_size

        self.inceptions = []
        self.shortcuts = []

        prev = feature_size
        residual_prev = prev

        for d in range(depth):
            self.inceptions.append(InceptionModule(
                prev,
                inception_filter,
                kernels,
                use_bottleneck,
                use_attn = use_attn,
            ))

            if use_residual and d % 3 == 2: # 2,5
                self.shortcuts.append(ResidualLayer(
                    input_dim = residual_prev,
                    output_dim = (len(kernels)+1) * inception_filter
                ))
                residual_prev = prev

            prev = (len(kernels) + 1) * inception_filter

        self.inceptions = nn.ModuleList(self.inceptions)
        self.shortcuts = nn.ModuleList(self.shortcuts)

        self.fcn = []
        for i in range(fcn_layers):
            self.fcn.append(FCNLayer(prev,fcn_filter,kernel_size=5,stide=2,padding=2))
            prev = fcn_filter
        
        self.fcn = nn.ModuleList(self.fcn)
        self.out = nn.Linear(fcn_filter,label_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

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

        x = self.fcn(x)
        x = torch.mean(x,dim=2) # NCL -> NC (average pooling)
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)

        return x
  
