from dataclasses import dataclass

import torch
from torch_timeseries.model import Crossformer
from src.experiments.shortterm.short_term import ShortTermTrafficExperiment
import fire


@dataclass
class CrossformerParameters:
    seg_len:int = 12 # following crossformer paper,segment length from 4 to 24, samller segment yield better results
    win_size:int = 2  # default:2 , since winsize 2 is used in crossformer original paper
    factor:int = 10
    d_model:int= 128
    d_ff:int = 256
    n_heads:int=2
    e_layers:int=3 
    dropout:float=0.2
    baseline = False

@dataclass
class CrossformerForecast(ShortTermTrafficExperiment, CrossformerParameters):
    model_type: str = "Crossformer"

    def _init_model(self):
        self.model = Crossformer(
            data_dim=self.dataset.num_features, in_len=self.windows, out_len=self.pred_len, seg_len=self.seg_len, win_size = self.win_size,
                factor=self.factor, d_model=self.d_model, d_ff = self.d_ff, n_heads=self.n_heads, e_layers=self.e_layers, 
                dropout=self.dropout, baseline =self.baseline, device=self.device
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

        outputs = self.model(batch_x)
            
        return outputs, batch_y


def cli():
    fire.Fire(CrossformerForecast)
    

if __name__ == "__main__":
    cli()
