from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import wandb
import time
import torch
from src.models.PAN_s import PAN
from src.datasets import *
from torch_timeseries.experiments import (
    ForecastExp,
)
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible


@dataclass
class PANarameters:
    start_d_model:int = 512
    d_c:int = 32
    d_patch:int = 128
    end_d_model : int = 512
    patch_len : int = 96
    stride : int = 48
    T_max : int = 8
    flatten : bool = False
    revin : bool = True
    cd_info : bool = True
    patch_info : bool = True
    raw_info:bool = True
    shared  : bool = True


@dataclass
class PANForecast(ForecastExp, PANarameters):
    model_type: str = "PAN_s"

    def _init_model(self):
        self.model = PAN(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            start_d_model=self.start_d_model,
            end_d_model=self.end_d_model,
            revin = self.revin,
            cd_info = self.cd_info,
            flatten=self.flatten,
            shared=self.shared,
            patch_len=self.patch_len,
            patch_info=self.patch_info,
            device=self.device,
            stride=self.stride,
            raw_info=self.raw_info
            )
        self.model = self.model.to(self.device)

    def _init_optimizer(self):
        super()._init_optimizer()
        if self.T_max != 0 :
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.model_optim, T_max=self.T_max, eta_min=0.00001
            )
        

    def run(self, seed=42) -> Dict[str, float]:

        if self._use_wandb() and not self._init_wandb(self.project, seed): return {}
        
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        if self._use_wandb():
            
            wandb.run.summary["parameters"] = model_parameters_num

        # for resumable reproducibility
        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            # for resumable reproducibility
            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            val_result = self._val()
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result[self.loss_func_type], model=self.model)

            self._save_run_check_point(seed)

            if self._use_wandb():
                wandb.log({'training_loss' : np.mean(train_losses)}, step=self.current_epoch)
                wandb.log( {f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                wandb.log( {f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)
            if self.T_max != 0:
                self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        if self._use_wandb():
            for k, v in best_test_result.items(): wandb.run.summary[f"best_test_{k}"] = v 
        
        if self._use_wandb():  wandb.finish()
        return best_test_result

        
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
    fire.Fire(PANForecast)