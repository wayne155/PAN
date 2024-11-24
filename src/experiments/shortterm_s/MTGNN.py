import numpy as np
import torch
from dataclasses import dataclass

import torch
from src.experiments.shortterm.short_term import ShortTermTrafficExperiment

from torch_timeseries.core.dataset import TimeSeriesStaticGraphDataset
from src.models.MTGNN.MTGNN import MTGNN
from src.utils.adj import adj_to_edge_index_weight, get_normalized_adj

from dataclasses import dataclass


@dataclass
class MTGNNExperiment(ShortTermTrafficExperiment):
    model_type: str = "MTGNN"
    gcn_true: bool = True
    gcn_depth: int = 2
    dropout: float = 0.3
    subgraph_size: int = 3
    node_dim: int = 40
    dilation_exponential: float = 2
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    in_dim: int = 1
    # out_dim:int=12
    layers: int = 5
    propalpha: float = 0.05
    tanhalpha: float = 3
    layer_norm_affline: bool = False
    skip_layer: bool = True
    residual_layer: bool = True

    invtrans_loss: bool = False

    def _init_model(self):
        assert (
            self.subgraph_size <= self.dataset.num_features
        ), f"graph size {self.subgraph_size} have to be small than data columns :{self.dataset.num_features}"
        predefined_A = None
        if self.pred_len > 1 and isinstance(self.dataset, TimeSeriesStaticGraphDataset):
            predefined_A = self.dataset.adj
            
        self.model = MTGNN(
            gcn_true=self.gcn_true,
            buildA_true=True,
            gcn_depth=self.gcn_depth,
            num_nodes=self.dataset.num_features,
            device=self.device,
            predefined_A=predefined_A,
            static_feat=None,
            dropout=self.dropout,
            subgraph_size=self.subgraph_size,
            node_dim=self.node_dim,
            dilation_exponential=self.dilation_exponential,
            conv_channels=self.conv_channels,
            residual_channels=self.residual_channels,
            skip_channels=self.skip_channels,
            end_channels=self.end_channels,
            seq_length=self.windows,
            in_dim=self.in_dim,
            out_dim=self.pred_len,
            layers=self.layers,
            propalpha=self.propalpha,
            tanhalpha=self.tanhalpha,
            layer_norm_affline=self.layer_norm_affline,
            skip_layer=self.skip_layer,
            residual_layer=self.residual_layer,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device)
        batch_y_date_enc = batch_y_date_enc.to(self.device)
        input_x = batch_x.unsqueeze(1)
        input_x = input_x.transpose(
            2, 3
        )  # torch.Size([batch_size, 1, num_nodes, windows])
        outputs = self.model(input_x)  # torch.Size([batch_size, seq_len, num_nodes, 1])
        outputs = outputs.squeeze(3)
        
        if self.pred_len == 1:
            return outputs.squeeze(1), batch_y.squeeze(1)
        elif self.pred_len > 1:
            return outputs, batch_y



def cli():
    import fire
    fire.Fire(MTGNNExperiment)

if __name__ == "__main__":
    cli()

