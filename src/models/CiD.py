import torch
from torch import nn
from torch_timeseries.nn.encoder  import Encoder, EncoderLayer
from torch_timeseries.nn.attention import FullAttention, AttentionLayer
from torch_timeseries.nn.embedding import PatchEmbedding


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
    def __init__(self,enc_in, in_size, mask=True, hidden_size=512, ti=95):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(in_size + enc_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, in_size)
        )
        self.ti = ti
        self.norm = nn.LayerNorm(in_size)
        self.mask = mask
        
    def forward(self, x, raw_x):
        # x : B, I, N
        # cx : B, 1, N
        
        B, T, N = x.shape
        lst = raw_x[:, self.ti, :].unsqueeze(1).repeat(1, N, 1)
        if self.mask:
            mask = (1 - torch.eye(N).unsqueeze(0)).to(x.device) # 1, N, N
            lst = lst * mask # B, N, N
        
        x = x.permute(0, 2, 1) # (B N I)
        inp = torch.concat([x, lst], dim=2) # B, N, N+I
        out = self.norm(x + self.linear(inp))
        return out.permute(0, 2, 1)
        
        
        
        
        
        
class CiD(nn.Module):
    def __init__(self, seq_len,pred_len,enc_in, mask=True, layer_ti=[0, 48 ,95], hidden_size=128,n_heads=8,dropout=0.0,e_layers=2,d_model=512,d_ff=512, patch_len=16, stride=8,task_name="long_term_forecast",num_class=0):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride
        
        self.hidden_size = hidden_size
        self.mask = mask

        # self.layers = layers
        
        
        self.embed = nn.Linear(seq_len, hidden_size)
        # self.start = CiDBlock(enc_in=enc_in, in_size=hidden_size, mask=mask, hidden_size=hidden_size)
        self.mid = nn.ModuleList(
            [CiDBlock(enc_in=enc_in, in_size=hidden_size, mask=mask, hidden_size=hidden_size, ti=i) for i in layer_ti]
        )
        # for i in range(len(layer_ti)):
        #     self.mid.append(CiDBlock(enc_in=enc_in, in_size=hidden_size, mask=mask, hidden_size=hidden_size))
            
        # self.end = CiDBlock(enc_in=enc_in, in_size=hidden_size, mask=mask, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        raw_x = x_enc
        B, T, N = x_enc.shape
        # x_enc : B T N

        #  B * N, (T+N)   B, N, N
        x_enc = self.embed(x_enc.transpose(1,2)).transpose(1,2)
        for block in self.mid:
            x_enc = block(x_enc, raw_x)
        # x = self.start(x_enc, x_enc[:, -1:, :])
        # x = self.end(x, x_enc[:, :1, :])
        dec_out = self.output(x_enc.transpose(1,2)).transpose(1,2)
        
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
