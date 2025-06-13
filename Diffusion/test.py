import torch
from torch import nn
import torch.optim as optim
import torch.functional as F
import math 


#Multi-head attention的基本参数
#batch time dimension
X=torch.randn(128,64,512)

#QKV    
d_model=512
num_heads=8

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        
        self.w_combine=nn.Linear(d_model,d_model)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,q,k,v):
        batch,time,dimension=q.shape
        n_d=self.d_model//self.num_heads
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)

        #三个维度，batch,head,time,dimension
        q=q.view(batch,time,self.num_heads,n_d).permute(0,2,1,3)
        k=k.view(batch,time,self.num_heads,n_d).permute(0,2,1,3)
        v=v.view(batch,time,self.num_heads,n_d).permute(0,2,1,3)

        score=q@k.transpose(2,3)/math.sqrt(n_d)
        mask=torch.tril(torch.ones(time,time,dtype=bool))
        #mask 是下三角矩阵，用于掩盖未来的信息,将mask中为0的元素填充为-inf
        score=score.masked_fill(mask==0,-float('inf'))
        score=self.softmax(score)@v

        score=score.permute(0,2,1,3).contiguous().view(batch,time,dimension)

        output=self.w_combine(score)
        return output
    
attention=MultiHeadAttention(d_model,num_heads)

output=attention(X,X,X)
print(output,output.shape)


class Encoder(nn.Module):
    def __init__(self, voc_size, max_len, d_model, n_heads, n_layers, drop_prob, device):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(voc_size, d_model)
        self.pos_encoding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, drop_prob) for _ in range(n_layers)])
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        pos = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len)
        x = self.embedding(x) + self.pos_encoding(pos)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x):
        attention = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward))
        return x

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, max_len, d_model, n_heads, ffn_hidden, n_layers, drop_prob, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, n_heads, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, n_heads, n_layers, drop_prob, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        
    def forward(self, src, trg):
        enc_output = self.encoder(src)
        dec_output = self.decoder(trg, enc_output)
        return dec_output

