from dataclasses import dataclass
import sys

import torch
from torch_timeseries.model import DLinear
import time
from src.experiments.shortterm.short_term import ShortTermTrafficExperiment
import fire
@dataclass
class DLinearParameters:
    individual: bool = False


@dataclass
class DLinearForecast(ShortTermTrafficExperiment, DLinearParameters):
    model_type: str = "DLinear"

    def _init_model(self):
        self.model = DLinear(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            individual=self.individual,
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
        outputs = self.model(
            batch_x
        )  # torch.Size([batch_size, output_length, num_nodes])
        return outputs, batch_y
    
    
def cli():
    fire.Fire(DLinearForecast)
    

if __name__ == "__main__":
    cli()
