from dataclasses import dataclass
import sys

import torch
from src.models.CPatchTST import CDPatchTST
from torch_timeseries.experiments import (
    ForecastExp,
)


@dataclass
class CDPatchTSTParameters:
    d_model: int = 512
    e_layers: int = 2
    d_ff: int = 512  # out of memoery with d_ff = 2048
    dropout: float = 0.0
    n_heads : int = 8
    patch_len : int = 16
    stride : int = 8
    label_len :int = 48


@dataclass
class CDPatchTSTForecast(ForecastExp, CDPatchTSTParameters):
    model_type: str = "CDPatchTST"

    def _init_model(self):
        self.model = CDPatchTST(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            dropout=self.dropout,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            task_name="long_term_forecast",
            num_class=0
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
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, -self.label_len:, :], batch_y_date_enc], dim=1
        )
        outputs = self.model(batch_x, batch_x_date_enc,
                             dec_inp, dec_inp_date_enc)
        return outputs, batch_y


if __name__ == "__main__":
    import fire
    fire.Fire(CDPatchTSTForecast)