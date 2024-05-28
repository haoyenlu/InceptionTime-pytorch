import torch
import torch.nn as nn

from layer import InceptionModule, ResidualLayer

from embedding_layer import DataEmbedding

from embedding_layer import PositionalEmbedding
from transformer_layer import Encoder


class InceptionTime(nn.Module):
    def __init__(self,batch_size,sequence_len,feature_size,label_dim, 
                filter_size=32,depth=6,kernels = [10,20,40],dropout=0.2,
                use_residual=True, use_bottleneck=True,use_attn=False,use_embedding=False):
        
        super(InceptionTime,self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sequence_len = sequence_len
        self.feature_size = feature_size
        self.label_dim = label_dim
        self.depth = depth
        self.use_residual = use_residual
        self.use_attn = use_attn
        self.use_embedding = use_embedding
        self.filter_size = filter_size
        self.batch_size = batch_size



        self.embedding = DataEmbedding(c_in = feature_size,d_model=feature_size,dropout=dropout,max_len=sequence_len)

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

        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(prev,hidden_size=filter_size,num_layers=4,batch_first=True)
        self.lstm_fn = nn.Linear(filter_size, label_dim)        

        self.out = nn.Sequential(
            nn.Linear(prev,label_dim * 2),
            nn.ReLU(),
            nn.Linear(label_dim * 2,label_dim),
        )

        self.softmax = nn.Softmax()

    def forward(self,x): # input shape: (N,C,L)
        assert self.sequence_len == x.shape[2] and self.feature_size == x.shape[1]
        
        if self.use_embedding:
            x = self.embedding(x.permute((0,2,1))).permute((0,2,1))
        

        res_input = x
        s_index = 0
        for d in range(self.depth):
            x = self.inceptions[d](x)

            if self.use_residual and d % 3 == 2:
                x = self.shortcuts[s_index](res_input,x)
                res_input = x
                s_index += 1

        # incep_out = torch.mean(x,dim=2) # NC
        lstm_out,  (_,_) = self.lstm(x.permute((0,2,1)), self.hidden) # NLC - > NLH

        x = self.lstm_fn(lstm_out[:,-1,:])
        # x = self.softmax(x)

        return x
  
    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(4, self.batch_size, self.filter_size),requires_grad=False).to(self.device),
                torch.autograd.Variable(torch.zeros(4, self.batch_size, self.filter_size),requires_grad=False).to(self.device))

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

    def forward(self,x): # NCL
        x = x.permute((0,2,1)) # NLC
        x = self.encoder_input_layer(x)
        x = self.emb(x)
        x = self.encoder(x)
        x = self.out(x)
        return x