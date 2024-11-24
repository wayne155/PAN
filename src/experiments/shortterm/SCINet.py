from dataclasses import dataclass
import sys

import torch
from torch_timeseries.model import SCINet
import time
from src.experiments.shortterm.short_term import ShortTermTrafficExperiment

@dataclass
class SCINetParameters:
    hid_size : int = 1
    num_stacks : int = 1
    num_levels : int = 3
    num_decoder_layer : int = 1
    concat_len : int = 0
    groups : int = 1
    kernel : int = 5
    dropout : float = 0.5
    single_step_output_One : int = 0
    input_len_seg : int = 1
    positionalE : bool = False
    modified : bool = True
    RIN : bool = True
    
@dataclass
class SCINetForecast(ShortTermTrafficExperiment, SCINetParameters):
    model_type: str = "SCINet"
        
    def _init_model(self):
        self.label_len = int(self.windows / 2)
        
        self.model = SCINet(
            output_len=self.pred_len, 
            input_len=self.windows, 
            input_dim = self.dataset.num_features,
            hid_size = self.hid_size, 
            num_stacks = self.num_stacks,
            num_levels = self.num_levels, 
            num_decoder_layer = self.num_decoder_layer, 
            concat_len = self.concat_len, 
            groups = self.groups, 
            kernel = self.kernel, 
            dropout = self.dropout,
            single_step_output_One = self.single_step_output_One, 
            input_len_seg = self.input_len_seg, 
            positionalE =self.positionalE, 
            modified = self.modified, 
            RIN=self.RIN
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
        
        pred = self.model(batch_x)
        
            
        return pred, batch_y



if __name__ == "__main__":
    import fire
    fire.Fire(SCINetForecast)