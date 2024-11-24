from dataclasses import dataclass, field
import sys

import torch
from src.models.CiDLPT import CiDP
from torch_timeseries.experiments import (
    ForecastExp,
)


@dataclass
class CiDParameters:
    d_model : int = 256
    d_ff : int = 512
    patch_len : int = 16
    stride : int = 8
    mask : bool = True
    flatten : bool = False


@dataclass
class CiDForecast(ForecastExp, CiDParameters):
    model_type: str = "CiDPLT"

    def _init_model(self):
        self.model = CiDP(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            d_ff=self.d_ff,
            flatten=self.flatten,
            mask=self.mask
            )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()

        # no decoder input
        # label_len = 1
        outputs = self.model(batch_x, None, None, None)
        return outputs, batch_y




if __name__ == "__main__":
    import fire
    fire.Fire(CiDForecast)