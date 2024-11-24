import torch
import torchmetrics

class SMAPE(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_absolute_percentage", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        smape = torch.abs(preds - target) / ((torch.abs(preds) + torch.abs(target)) / 2)
        self.sum_absolute_percentage += torch.sum(smape)
        self.total += target.numel()

    def compute(self):
        return 200.0 * self.sum_absolute_percentage / self.total


class MASE(torchmetrics.Metric):
    def __init__(self, seasonality: int = 1):
        super().__init__()
        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("naive_error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.seasonality = seasonality

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        error = torch.abs(preds - target)
        naive_error = torch.abs(target[self.seasonality:] - target[:-self.seasonality])
        self.error_sum += torch.sum(error)
        self.naive_error_sum += torch.sum(naive_error)
        self.total += target.numel()

    def compute(self):
        return self.error_sum / (self.total * (self.naive_error_sum / (self.total - self.seasonality)))


class OWA(torchmetrics.Metric):
    def __init__(self, seasonality: int = 1):
        super().__init__()
        self.smape_metric = SMAPE()
        self.mase_metric = MASE(seasonality=seasonality)
        self.add_state("smape_naive2", default=torch.tensor(1.0))  # Set default as needed
        self.add_state("mase_naive2", default=torch.tensor(1.0))   # Set default as needed

    def update(self, preds: torch.Tensor, target: torch.Tensor, smape_naive2: float, mase_naive2: float):
        self.smape_naive2 = torch.tensor(smape_naive2)
        self.mase_naive2 = torch.tensor(mase_naive2)
        self.smape_metric.update(preds, target)
        self.mase_metric.update(preds, target)

    def compute(self):
        smape = self.smape_metric.compute()
        mase = self.mase_metric.compute()
        owa = 0.5 * (smape / self.smape_naive2 + mase / self.mase_naive2)
        return owa
