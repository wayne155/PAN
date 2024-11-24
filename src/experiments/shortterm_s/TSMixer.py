from dataclasses import dataclass
import sys

import torch
from torch_timeseries.model import TSMixer
import time
from src.experiments.shortterm.short_term import ShortTermTrafficExperiment

@dataclass
class TSMixerParameters:
    n_mixer : int = 8
    dropout :int  = 0.05

@dataclass
class TSMixerForecast(ShortTermTrafficExperiment, TSMixerParameters):
    model_type: str = "TSMixer"
        
    
    def _init_model(self):
        self.model = TSMixer(
            L=self.windows,
            C = self.dataset.num_features,
            T=self.pred_len,
            n_mixer=self.n_mixer,
            dropout=self.dropout
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
        
        outputs = self.model(batch_x)  # torch.Size([batch_size, output_length, num_nodes])
            
        return outputs, batch_y




if __name__ == "__main__":
    import fire
    fire.Fire(TSMixerForecast)