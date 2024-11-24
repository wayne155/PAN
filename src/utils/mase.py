import torch
import torchmetrics

class MASE(torchmetrics.Metric):
    def __init__(self, seasonality: int = 1):
        super().__init__()
        # State variables to accumulate error sums
        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("naive_error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.seasonality = seasonality

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate the absolute error between predictions and targets
        error = torch.abs(preds - target)
        self.error_sum += torch.sum(error)

        # Calculate the naive method's absolute error
        if len(target) > self.seasonality:
            naive_error = torch.abs(target[self.seasonality:] - target[:-self.seasonality])
            self.naive_error_sum += torch.sum(naive_error)

        self.n += target.numel()  # Count the number of target elements

    def compute(self):
        # Calculate MASE
        if self.n == 0 or self.naive_error_sum == 0:
            return torch.tensor(float('inf'))  # Avoid division by zero
        mase = self.error_sum / (self.naive_error_sum / (self.n - self.seasonality))
        return mase
