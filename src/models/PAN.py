import torch
from torch import nn
from torch_timeseries.nn.encoder  import Encoder, EncoderLayer
from torch_timeseries.nn.attention import FullAttention, AttentionLayer

def generate_even_numbers(start_d_model, end_d_model, k):
    if start_d_model % 2 != 0:
        start_d_model += 1
    if end_d_model % 2 != 0:
        end_d_model -= 1

    numbers = torch.linspace(start_d_model, end_d_model, steps=k)
    
    even_numbers = (torch.round(numbers / 2) * 2).int()
    
    return even_numbers

class PatchAmplifier(nn.Module):
    def __init__(self, enc_in, patch_emb, in_size, device, patch_info=True, cd_info=True, out_size=512):
        super().__init__()
        
        l_in_size = ((patch_emb if patch_info else 0) + in_size  + (enc_in if cd_info else 0))
        self.linear = nn.Sequential(
            nn.Linear(l_in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size)
        )
        self.cd_info = cd_info
        self.patch_info = patch_info
        self.skip = nn.Linear(in_size, out_size)
        self.norm = nn.LayerNorm(int(out_size))

        
    def forward(self, patch, prev, ceb):
        # patch : B, N, patch_emb
        # prev : B, N, d_model
        
        B, N, D = patch.shape
        
        lst = ceb.unsqueeze(1).repeat(1, N, 1)
        if self.cd_info and self.patch_info:
            inp = torch.concat([prev, patch, lst], dim=2) # B, N, N+I
        elif self.cd_info:
            inp = torch.concat([prev, lst], dim=2) # B, N, N+I
        elif self.patch_info:
            inp = torch.concat([prev, patch], dim=2) # B, N, N+I
        else:
            inp = torch.concat([prev], dim=2) # B, N, N+I
            
        out = self.norm(self.skip(prev) + self.linear(inp))
        return out
        
        
        
        
        
        
class PAN(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, device, raw_info=True, flatten=False, cd_info=True, patch_info=True, revin=False, d_c=7, d_patch=128, hidden_size=128,start_d_model=128,end_d_model=512 ,dropout=0.0, patch_len=96, stride=48, task_name="long_term_forecast"):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride
        self.raw_info = raw_info
        self.patch_len = patch_len
        self.patch_num = int((seq_len - patch_len) / stride + 2)
        self.task_name = "long_term_forecast"
        self.hidden_size = hidden_size
        self.device = device
        self.patch_info =patch_info
        self.flatten = flatten
        # self.d_model = d_model
        self.stride = stride
        self.cd_info = cd_info
        self.revin =revin
        if start_d_model == end_d_model:
            self.d_models = [start_d_model]*(self.patch_num+1)
        else:
            self.d_models = generate_even_numbers(start_d_model, end_d_model, k=self.patch_num+1)
        print(self.d_models)
        print(self.patch_num)
        self.padding_patch_layer = nn.ReplicationPad1d((padding, 0))
        
        self.value_embedding = nn.Linear(patch_len, d_c, bias=False)
        self.channel_embedding = nn.Linear(enc_in, d_patch, bias=False)
        self.d_c = d_c
        self.d_patch = d_patch
        self.dropout = nn.Dropout(dropout)
        # self.layers = layers
        self.t_index = [i for i in range(self.patch_len -1, seq_len+stride, stride)]
        print("self.t_index", self.t_index)
        assert len(self.t_index) == self.patch_num
        
        self.init_embedding = nn.Parameter(torch.randn((1, enc_in, self.d_models[0])))
        if self.raw_info:
            print("raw_info..")
            self.mid = nn.ModuleList(
                [PatchAmplifier(enc_in=enc_in, patch_emb=patch_len, patch_info=patch_info, cd_info=self.cd_info, device=self.device, in_size=self.d_models[i], out_size=self.d_models[i+1]) for i, _ in enumerate(self.t_index)]
            )
        else:
            print("not raw_info..")
            self.mid = nn.ModuleList(
                [PatchAmplifier(enc_in=d_c, patch_emb=d_patch, patch_info=patch_info, cd_info=self.cd_info, device=self.device, in_size=self.d_models[i], out_size=self.d_models[i+1]) for i, _ in enumerate(self.t_index)]
            )
        if self.flatten:
            self.output = nn.Linear(sum(self.d_models[1:]), pred_len)
        else:
            self.output = nn.Linear(self.d_models[-1], pred_len)
            

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.revin:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
        B, T, N = x_enc.shape 
        
        
        if self.raw_info:
            # padding
            x_enc = x_enc.permute(0, 2, 1) # B N T
            x_enc = self.padding_patch_layer(x_enc)
            raw_x = x_enc # B N T+pad 
            # patch embedding
            x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            x_enc = torch.reshape(x_enc, (B, N, x_enc.shape[2], x_enc.shape[3]))

            init = self.init_embedding.repeat(B, 1, 1) # B, N, d_model torch.zeros_like(B, N, self.d_model).to(patch_enc.device)
            prevs = []
            for i, block in enumerate(self.mid):
                if i == 0:
                    prev = block(x_enc[:,:, i, :] , init, raw_x[:, :, self.t_index[i]]) # B*N d_model
                else:
                    prev = block(x_enc[:,:, i, :], prev, raw_x[:, :, self.t_index[i]]) # B*N d_model
                prevs.append(prev)
        else:
            x_enc = x_enc.permute(0, 2, 1) # B N T
            x_enc = self.padding_patch_layer(x_enc)
            channel_emb = self.channel_embedding(x_enc.transpose(1,2))
            x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            x_enc = torch.reshape(x_enc, (B, N, x_enc.shape[2], x_enc.shape[3]))
            x_enc = self.value_embedding(x_enc)
            init = self.init_embedding.repeat(B, 1, 1) # B, N, d_model torch.zeros_like(B, N, self.d_model).to(patch_enc.device)
            prevs = []
            for i, block in enumerate(self.mid):
                if i == 0:
                    prev = block(x_enc[:,:, i, :] , init, channel_emb[:, self.t_index[i], :]) # B N d_model
                else:
                    prev = block(x_enc[:,:, i, :], prev, channel_emb[:, self.t_index[i], :]) # B N d_model
                prevs.append(prev)
        self.prevs = prevs
        if self.flatten:   
            prevs = torch.concat(prevs, dim=-1)     
            dec_out = self.output(prevs)
        else:
            dec_out = self.output(prev)
        dec_out = dec_out.permute(0, 2, 1 )
        
        if self.revin:
            #De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
