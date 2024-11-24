import torch
from torch import nn
from torch_timeseries.nn.encoder  import Encoder, EncoderLayer
from torch_timeseries.nn.attention import FullAttention, AttentionLayer
"""
this version testing
1. wether embed prev increasing the results or not?
"""
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((padding, 0))

        
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) 
        return self.dropout(x), n_vars
    
    
    
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class CiDBlock(nn.Module):
    def __init__(self,enc_in, patch_emb, d_model, mask=True, hidden_size=512, ti=95):
        super().__init__()
        
        # self.linear = nn.Sequential(
        #     nn.Linear(patch_emb, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, d_model)
        # )
        self.ti = ti
        self.norm = nn.LayerNorm(d_model)
        self.mask = mask
        self.dropout = nn.Dropout()
        
    def forward(self, patch, prev, raw_x):
        # patch : B, N, patch_emb
        # prev : B, N, d_model
        
        # raw_x : B, N, T
        
        B, N, D = patch.shape
        
        # lst = raw_x[:, : , self.ti].unsqueeze(1).repeat(1, N, 1)
        # if self.mask:
        #     mask = (1 - torch.eye(N).unsqueeze(0)).to(patch.device) # 1, N, N
        #     lst = lst * mask # B, N, N
        
        # x = x.permute(0, 2, 1) # (B N I)
        # inp = torch.concat([patch], dim=2) # B, N, N+I
        out = self.norm(prev + patch)
        # out = self.norm(prev + #self.linear(patch))
        return out
        
        
        
        
        
        
class CiDP(nn.Module):
    def __init__(self, seq_len,pred_len,enc_in, mask=True, hidden_size=128 , flatten=True,dropout=0.0,d_model=256,d_ff=512, patch_len=24, stride=16, task_name="long_term_forecast"):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride
        self.patch_len = patch_len
        self.patch_num = int((seq_len - patch_len) / stride + 2)
        self.task_name = "long_term_forecast"
        self.hidden_size = hidden_size
        self.mask = mask
        self.d_model = d_model
        self.stride = stride
        self.flatten = flatten
        
        
        self.padding_patch_layer = nn.ReplicationPad1d((padding, 0))
        
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # self.layers = layers
        self.t_index = [i for i in range(self.patch_len -1, seq_len+stride, stride)]
        print("self.t_index", self.t_index)
        assert len(self.t_index) == self.patch_num
        
        
        self.patch_weight1 = nn.Parameter(torch.randn(1, self.patch_num, d_model, d_model))
        self.patch_bias1 = nn.Parameter(torch.randn(1, self.patch_num, d_model))
        
        self.patch_weight2 = nn.Parameter(torch.randn(1, self.patch_num, d_model, d_model))
        self.patch_bias2 = nn.Parameter(torch.randn(1, self.patch_num, d_model))
    
        
        self.init_embedding = nn.Parameter(torch.randn((1, enc_in, d_model)))
        
        self.mid = nn.ModuleList(
            [CiDBlock(enc_in=enc_in, patch_emb=d_model, d_model=d_model, mask=mask, hidden_size=d_ff, ti=i) for i in self.t_index]
        )
        if self.flatten:
            self.output = nn.Linear(self.patch_num*d_model, pred_len)
        else:
            self.output = nn.Linear(d_model, pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        B, T, N = x_enc.shape 
        
        
        # padding
        x_enc = x_enc.permute(0, 2, 1) # B N T
        x_enc = self.padding_patch_layer(x_enc)
        raw_x = x_enc # B N T+pad 

        # patch embedding
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_enc = torch.reshape(x_enc, (B, N, x_enc.shape[2], x_enc.shape[3]))
        x_enc = self.dropout(self.value_embedding(x_enc))  #  B, N, patch_num, d_model
        
        
        
        x_enc = torch.matmul(self.patch_weight1, x_enc.unsqueeze(-1)).squeeze(-1) + self.patch_bias1
        x_enc = torch.relu(x_enc)
        x_enc = torch.matmul(self.patch_weight2, x_enc.unsqueeze(-1)).squeeze(-1) + self.patch_bias2
        
        # g = 50
        # for i in range(0, N, g):
        #     x_enc[:, i:i+g, ...] = torch.matmul(self.patch_weight1, x_enc[:, i:i+g, ...].unsqueeze(-1)).squeeze(-1) + self.patch_bias1
        #     x_enc[:, i:i+g, ...] = torch.relu(x_enc[:, i:i+g, ...])
        #     x_enc[:, i:i+g, ...] = torch.matmul(self.patch_weight2, x_enc[:, i:i+g, ...].unsqueeze(-1)).squeeze(-1) + self.patch_bias2
        


        
        # u: [bs * nvars x patch_num x d_model]
        # patch_enc, n_vars = self.channel_embedding(x_enc)
        
        # B,N,pn,d -> B,N,pn,d
        
        #  B * N, (T+N)   B, N, N
        # x_enc = self.embed(x_enc.transpose(1,2)).transpose(1,2)
        init = self.init_embedding.repeat(B, 1, 1) # B, N, d_model torch.zeros_like(B, N, self.d_model).to(patch_enc.device)
        prevs = []
        for i, block in enumerate(self.mid):
            if i == 0:
                prev = block(x_enc[:,:, i, :] , init, raw_x) # B*N d_model
            else:
                prev = block(x_enc[:,:, i, :], prev, raw_x) # B, N d_model
                
            prevs.append(prev)
            
        # x = self.start(x_enc, x_enc[:, -1:, :])
        # x = self.end(x, x_enc[:, :1, :])
        if self.flatten:
            out = torch.concat(prevs, dim=2)
            dec_out = self.output(out)
        else:
            dec_out = self.output(prev)
            
        dec_out = dec_out.permute(0, 2, 1 )
        
        # # mask = 
        # lst = x_enc[:, -1:, :].repeat(1,  N, 1) # B, N, N 
        # if self.mask:
        #     mask = (1 - torch.eye(N).unsqueeze(0)).to(x_enc.device) # 1, N, N
        #     lst = lst * mask # B, N, N
        
        # x_enc = x_enc.permute(0, 2, 1) # (B N T)
        # inp = torch.concat([lst, x_enc], dim=2) # B, N, N+T
        
        # out = self.linear(inp) # B, N, O
        # dec_out = out.permute(0, 2, 1) # B O N
        
        # # do patching and embedding
        # # u: [bs * nvars x patch_num x d_model]
        # enc_out, n_vars = self.patch_embedding(x_enc)

        # # Encoder
        # # z: [bs * nvars x patch_num x d_model]
        # enc_out, attns = self.encoder(enc_out)
        # # z: [bs x nvars x patch_num x d_model]
        # enc_out = torch.reshape(
        #     enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # # z: [bs x nvars x d_model x patch_num]
        # enc_out = enc_out.permute(0, 1, 3, 2)
        
        

        # # Decoder
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)

        #De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)


        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        
        
        
        # # do patching and embedding
        # x_enc = x_enc.permute(0, 2, 1)
        # # u: [bs * nvars x patch_num x d_model]
        # enc_out, n_vars = self.patch_embedding(x_enc)

        # # Encoder
        # # z: [bs * nvars x patch_num x d_model]
        # enc_out, attns = self.encoder(enc_out)
        # # z: [bs x nvars x patch_num x d_model]
        # enc_out = torch.reshape(
        #     enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # # z: [bs x nvars x d_model x patch_num]
        # enc_out = enc_out.permute(0, 1, 3, 2)

        # # Decoder
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
