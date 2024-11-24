import torch
from dataclasses import dataclass

import torch
from torch_timeseries.experiments import (
    ForecastExp,
)
from torch_timeseries.core.dataset import TimeSeriesStaticGraphDataset
from torch_timeseries.utils.parse_type import parse_type
from torchmetrics import MeanAbsoluteError, MetricCollection, MeanSquaredError, SymmetricMeanAbsolutePercentageError
from src.datasets import *

from src.utils.owa import OWA
from src.utils.mase import MASE

from dataclasses import dataclass


@dataclass
class ShortTermTrafficExperiment(ForecastExp):
    
    def _init_dataset(self):
        self.dataset = parse_type(self.dataset_type, globals())(
            root=self.data_path
        )
            
    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "rmse": MeanSquaredError(squared=False),
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(),
                "smape": SymmetricMeanAbsolutePercentageError(),
            }
        )
        self.metrics.to(self.device)
        