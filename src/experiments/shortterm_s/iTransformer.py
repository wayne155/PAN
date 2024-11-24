from dataclasses import dataclass
import sys

import numpy as np
import torch
import wandb
from torch_timeseries.model import iTransformer
import time
from src.experiments.shortterm.short_term import ShortTermTrafficExperiment

from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible



@dataclass
class iTransformerParameters:
    factor: int = 1
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 4
    d_layer: int = 1
    d_ff: int = 512
    dropout: float = 0.1
    embed: str = "timeF"
    activation:str= "gelu"
    use_norm : bool = True
    class_strategy : str = 'projection'

@dataclass
class iTransformerForecast(ShortTermTrafficExperiment, iTransformerParameters):
    model_type: str = "iTransformer"
        
    def _init_model(self):
        self.label_len = int(self.windows / 2)
        
        self.model = iTransformer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            use_norm=self.use_norm,
            class_strategy=self.class_strategy,
            factor=self.factor,
            freq=self.dataset.freq,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            embed=self.embed,
            activation=self.activation,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len :, :], batch_y_date_enc], dim=1
        )
        outputs = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)  # torch.Size([batch_size, output_length, num_nodes])
            
        return outputs, batch_y

def cli():
    import fire
    fire.Fire(iTransformerForecast)


if __name__ == "__main__":
    cli()
