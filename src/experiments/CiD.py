from dataclasses import dataclass, field
import sys

import torch
from src.models.CiD import CiD
from torch_timeseries.experiments import (
    ForecastExp,
)


@dataclass
class CiDParameters:
    hidden_size : int = 256
    mask : bool = True
    layer_ti : list = field(default_factory= lambda : [0, 48, 95])


@dataclass
class CiDForecast(ForecastExp, CiDParameters):
    model_type: str = "CiD"

    def _init_model(self):
        self.model = CiD(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            hidden_size=self.hidden_size,
            layer_ti=self.layer_ti,
            # layer_ti=[0, 10, 20, 30, 40, 50, 60, 70, 80, 95],
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