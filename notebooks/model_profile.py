
import os
import sys
import torch
import time
import numpy as np
sys.path.insert(0,os.path.abspath('/notebooks/4901_revisit_cdtran'))
sys.path.insert(0,os.path.abspath('/notebooks/pytorchtimseries'))
from tqdm.notebook import tqdm
from torch_timeseries.dataset import *
from torch_timeseries.experiments import *
# from src.experiments.shortterm import *

# from src.experiments.CiDPG import CiDPGForecast
from src.experiments.PAN import PANForecast
# from src.experiments.shortterm.PAN import PANForecast
from src.datasets import *
from torch_timeseries.utils.model_stats import count_parameters
import pandas as pd

# from src.experiments.iTransformer import iTransformerExp
# exp = DLinearForecast(data_path='/notebooks/4901_revisit_cdtran/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
# exp = DLinearForecast(dataset_type="ExchangeRate", data_path='/notebooks/pytorchtimseries/data', save_dir='/notebooks/pytorchtimseries/results', device='cuda:0')
# exp = PatchTSTForecast(dataset_type="ExchangeRate", data_path='/notebooks/pytorchtimseries/data', save_dir='/notebooks/pytorchtimseries/results', device='cuda:0')

def pan_profile( dataset_type, windows, pred_len, start_d_model, end_d_model, patch_len, stride ):
    model_type = "PAN"
    # exp = PANForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/pytorchtimseries/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
    # exp = PANForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/4901_revisit_cdtran/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
    exp = PANForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/4901_revisit_cdtran/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
    exp._setup_run(1000)
    memories = []
    
    from torch_timeseries.utils.model_stats import count_parameters
    _, nparam = count_parameters(exp.model)
    torch.cuda.empty_cache()
    steps = 20
    i = 0
    self = exp
    self.model.train()
    t = 0
    for i, (
        batch_x,
        batch_y,
        origin_x,
        origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(self.train_loader):
        if i >= steps:
            break
        i+= 1
        # print(i)
        origin_y = origin_y.to(self.device).float()
        self.model_optim.zero_grad()
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start = time.time()
        
        pred, true = self._process_one_batch(
            batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
        )
        loss = self.loss_func(pred, true)
        loss.backward()
        t += time.time() - start
        end_memory = torch.cuda.max_memory_allocated()
        memories.append(end_memory - start_memory)
        print(end_memory - start_memory)
        # print(f"t : {time.time() - start}")

        

    memories = np.array(memories)

    print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
    print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
    return "PAN", dataset_type, max(memories), min(memories), t/i*1000, nparam

def baseline_profile(model_type, dataset_type, windows, pred_len):

    exp = eval(f"{model_type}Forecast", globals())(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, data_path='/notebooks/4901_revisit_cdtran/data', device='cuda:0')
    exp._setup_run(1000)
    memories = []
    
    from torch_timeseries.utils.model_stats import count_parameters
    _, nparam = count_parameters(exp.model)
    torch.cuda.empty_cache()
    steps = 20
    i = 0
    self = exp
    self.model.train()
    t = 0
    for i, (
        batch_x,
        batch_y,
        origin_x,
        origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(self.train_loader):
        if i >= steps:
            break
        i+= 1
        origin_y = origin_y.to(self.device).float()
        self.model_optim.zero_grad()
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start = time.time()
        
        pred, true = self._process_one_batch(
            batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
        )
        loss = self.loss_func(pred, true)
        loss.backward()
        t += time.time() - start
        end_memory = torch.cuda.max_memory_allocated()
        memories.append(end_memory - start_memory)
        # print(f"t : {time.time() - start}")

        

    memories = np.array(memories)

    print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
    print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
    return max(memories), min(memories), t/i*1000, nparam



def traffic_baseline_profile(model_type, dataset_type, windows, pred_len):
    exp = eval(f"{model_type}Forecast", globals())(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, data_path='/notebooks/4901_revisit_cdtran/data', device='cuda:0')
    exp._setup_run(1000)
    memories = []
    
    from torch_timeseries.utils.model_stats import count_parameters
    _, nparam = count_parameters(exp.model)
    torch.cuda.empty_cache()
    steps = 20
    i = 0
    self = exp
    self.model.train()
    t = 0
    for i, (
        batch_x,
        batch_y,
        origin_x,
        origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(self.train_loader):
        if i >= steps:
            break
        i+= 1
        origin_y = origin_y.to(self.device).float()
        self.model_optim.zero_grad()
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start = time.time()
        
        pred, true = self._process_one_batch(
            batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
        )
        loss = self.loss_func(pred, true)
        loss.backward()
        t += time.time() - start
        end_memory = torch.cuda.max_memory_allocated()
        memories.append(end_memory - start_memory)
        # print(f"t : {time.time() - start}")

        

    memories = np.array(memories)

    print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
    print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
    return max(memories), min(memories), t/i*1000, nparam



# def profile_main_all():
#     models = ["PatchTST", "iTransformer", "Crossformer", "CATS", "DLinear", "Informer", "TSMixer", "SCINet"]
#     # models = ["Informer, "Crossformer""]
#     datasets = ["Traffic","ETTh1","ETTh2","ETTm1","ETTm2","Electricity","ExchangeRate","Weather"]
#     # datasets = ["ETTh1","ETTh2"]
#     # datasets = ["PEMS04","PEMS07","PEMS08","PEMS_BAY"]
#     all_data = []
#     for m in models:
#         for d in datasets:
#             mem, tim, nparam = baseline_profile(m, d, 336, 720)
#             all_data.append((m,d,mem, tim, nparam))
#         df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "traintime", "parameters"])
#         df.to_csv('profile.csv')

#     print(all_data)
#     df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "traintime", "parameters"])
#     df.to_csv('profile.csv')

def traffic_profile_main_all():
    
    models = ["PatchTST", "iTransformer", "Crossformer", "CATS", "DLinear", "Informer", "TSMixer", "SCINet"]
    models = ["PatchTST"]
    datasets = ["PEMS07", "PEMS04","PEMS08","PEMS_BAY"]
    all_data = []
    for m in models:
        for d in datasets:
            mem, minme, tim, nparam = traffic_baseline_profile(m, d, 96, 12)
            all_data.append((m,d,mem, minme, tim, nparam))
        df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "min_memory", "traintime", "parameters"])
        df.to_csv('traffic_profile.csv')

    print(all_data)
    df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "min_memory", "traintime", "parameters"])
    df.to_csv('traffic_profile.csv')
    

def pan_al_profile( dataset_type, windows, pred_len, start_d_model, end_d_model, patch_len, stride ):
    model_type = "PAN"
    assert start_d_model == end_d_model
    # exp = PANForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/pytorchtimseries/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
    exp = PANForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/4901_revisit_cdtran/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
    exp._setup_run(1000)
    memories = []
    
    from torch_timeseries.utils.model_stats import count_parameters
    _, nparam = count_parameters(exp.model)
    torch.cuda.empty_cache()
    steps = 20
    i = 0
    self = exp
    self.model.train()
    t = 0
    for i, (
        batch_x,
        batch_y,
        origin_x,
        origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(self.train_loader):
        if i >= steps:
            break
        i+= 1
        # print(i)
        origin_y = origin_y.to(self.device).float()
        self.model_optim.zero_grad()
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start = time.time()
        
        pred, true = self._process_one_batch(
            batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
        )
        loss = self.loss_func(pred, true)
        loss.backward()
        t += time.time() - start
        end_memory = torch.cuda.max_memory_allocated()
        memories.append(end_memory - start_memory)
        print(end_memory - start_memory)
        # print(f"t : {time.time() - start}")

        

    memories = np.array(memories)

    print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
    print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
    return "PAN", dataset_type, max(memories), min(memories), t/i*1000, nparam


def pan_lpo_profile( dataset_type, windows, pred_len, start_d_model, end_d_model, patch_len, stride ):
    model_type = "PAN"
    # exp = PANForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/pytorchtimseries/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
    exp = PANForecast(batch_size=1, dataset_type=dataset_type, flatten=True, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/4901_revisit_cdtran/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
    exp._setup_run(1000)
    memories = []
    
    from torch_timeseries.utils.model_stats import count_parameters
    _, nparam = count_parameters(exp.model)
    torch.cuda.empty_cache()
    steps = 20
    i = 0
    self = exp
    self.model.train()
    t = 0
    for i, (
        batch_x,
        batch_y,
        origin_x,
        origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(self.train_loader):
        if i >= steps:
            break
        i+= 1
        # print(i)
        origin_y = origin_y.to(self.device).float()
        self.model_optim.zero_grad()
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start = time.time()
        
        pred, true = self._process_one_batch(
            batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
        )
        loss = self.loss_func(pred, true)
        loss.backward()
        t += time.time() - start
        end_memory = torch.cuda.max_memory_allocated()
        memories.append(end_memory - start_memory)
        print(end_memory - start_memory)
        # print(f"t : {time.time() - start}")

        

    memories = np.array(memories)

    print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
    print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
    return "PAN", dataset_type, max(memories), min(memories), t/i*1000, nparam

    
def profile_main_all():
    models = ["PatchTST", "iTransformer", "Crossformer", "CATS", "DLinear", "Informer", "TSMixer", "SCINet"]
    models = ["PatchTST", "iTransformer", "Crossformer", "CATS", "DLinear", "Informer", "TSMixer", "SCINet"]
    # models = ["Informer, "Crossformer""]
    datasets = ["Traffic","ETTh1","ETTh2","ETTm1","ETTm2","Electricity","ExchangeRate","Weather"]
    # datasets = ["ETTh1","ETTh2"]
    # datasets = ["PEMS04","PEMS07","PEMS08","PEMS_BAY"]
    all_data = []
    for m in models:
        for d in datasets:
            mem, minme, tim, nparam = baseline_profile(m, d, 336, 720)
            all_data.append((m,d,mem, minme, tim, nparam))
        df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "min_memory", "traintime", "parameters"])
        df.to_csv('profile.csv')

    print(all_data)
    df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "min_memory", "traintime", "parameters"])
    df.to_csv('profile.csv')


def profile_main_all_windows():
    # models = ["PatchTST", "iTransformer", "Crossformer", "CATS", "DLinear", "Informer", "TSMixer", "SCINet"]
    models = ["PatchTST", "CATS"]
    # models = ["Informer, "Crossformer""]
    datasets = ["Traffic","ETTh1","ETTh2","ETTm1","ETTm2","Electricity","ExchangeRate","Weather"]
    datasets = ["ETTh1"]
    # datasets = ["PEMS04","PEMS07","PEMS08","PEMS_BAY"]
    all_data = []
    windows=[336, 512, 1024, 1536, 2048]

    d  = "ETTh1"
    for w in windows:
        for m in models:
            mem, minme, tim, nparam = baseline_profile(m,d, w, 720)
            all_data.append((m,d,mem, minme, tim, nparam))
            df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "min_memory", "traintime", "parameters"])
            df.to_csv('etth1_window_profile.csv')

    print(all_data)
    df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "min_memory", "traintime", "parameters"])
    df.to_csv('electricity_window_profile.csv')


def profile_pan_windows():
    # for w in windows:
    windows=[ 336, 512, 1024, 1536, 2048]
    all_data = [ pan_profile("ETTh2", w, 720, 128, 512, 96, 48) for w in windows ]
    
    # [
    #     pan_profile("ETTh1", 336, 720, 128, 512, 192, 48),
    #     # pan_profile("ETTh2", 336, 720, 128, 512, 96, 48),
    #     # pan_profile("ETTm1", 336, 720, 128, 512, 96, 48),
    #     # pan_profile("ETTm2", 336, 720, 128, 512, 96, 48),
    #     # pan_profile("Traffic", 336, 720, 128, 512, 192, 48),
    #     # pan_profile("ExchangeRate", 336, 720, 128, 512, 96, 48),
    #     # pan_profile("Weather", 336, 720, 128, 512, 96, 48),
    #     # pan_profile("Electricity", 336, 720, 128, 512, 288, 48),
    # ]
    print(all_data)
    df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "lowmem", "traintime", "parameters"])
    df.to_csv('pan_profile_windows.csv')
    
    



def profile_al_lpo():
    # for w in windows:
    # all_data = [
    #     pan_al_profile("ETTh1", 336, 720, 512, 512, 192, 48),
    #     pan_al_profile("ETTh2", 336, 720, 512, 512, 96, 48),
    #     pan_al_profile("ETTm1", 336, 720, 512, 512, 96, 48),
    #     pan_al_profile("ETTm2", 336, 720, 512, 512, 96, 48),
    #     pan_al_profile("Traffic", 336, 720, 512, 512, 192, 48),
    #     pan_al_profile("ExchangeRate", 336, 720, 512, 512, 96, 48),
    #     pan_al_profile("Weather", 336, 720, 512, 512, 96, 48),
    #     pan_al_profile("Electricity", 336, 720, 512, 512, 288, 48),
    # ]
    # print(all_data)
    # df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "lowmem", "traintime", "parameters"])
    # df.to_csv('pan_al_profile_windows.csv')
    
    
    all_data = [
        pan_lpo_profile("ETTh1", 336, 720, 128, 512, 192, 48),
        pan_lpo_profile("ETTh2", 336, 720, 128, 512, 96, 48),
        pan_lpo_profile("ETTm1", 336, 720, 128, 512, 96, 48),
        pan_lpo_profile("ETTm2", 336, 720, 128, 512, 96, 48),
        pan_lpo_profile("Traffic", 336, 720, 128, 512, 192, 48),
        pan_lpo_profile("ExchangeRate", 336, 720, 128, 512, 96, 48),
        pan_lpo_profile("Weather", 336, 720, 128, 512, 96, 48),
        pan_lpo_profile("Electricity", 336, 720, 128, 512, 288, 48),
    ]
    print(all_data)
    df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "lowmem", "traintime", "parameters"])
    df.to_csv('pan_lpo_profile_windows.csv')
    
    
def profile_pan_all():
    
    
    # etth1 = pan_profile("ETTh1", 336, 720, 128, 512, 192, 48)
    # etth2 = pan_profile("ETTh2", 336, 720, 128, 512, 96, 48)
    # ettm1 = pan_profile("ETTm1", 336, 720, 128, 512, 96, 48)
    # ettm2 = pan_profile("ETTm2", 336, 720, 128, 512, 96, 48)
    # traf= pan_profile("Traffic", 336, 720, 128, 512, 192, 48)
    # exg = pan_profile("ExchangeRate", 336, 720, 128, 512, 96, 48)
    # weather = pan_profile("Weather", 336, 720, 128, 512, 96, 48)
    # elec = pan_profile("Electricity", 336, 720, 128, 512, 288, 48)
    all_data = [
        # pan_profile("ETTh1", 336, 720, 128, 512, 192, 48),
        # pan_profile("ETTh2", 336, 720, 128, 512, 96, 48),
        # pan_profile("ETTm1", 336, 720, 128, 512, 96, 48),
        # pan_profile("ETTm2", 336, 720, 128, 512, 96, 48),
        # pan_profile("Traffic", 336, 720, 128, 512, 192, 48),
        # pan_profile("ExchangeRate", 336, 720, 128, 512, 96, 48),
        # pan_profile("Weather", 336, 720, 128, 512, 96, 48),
        # pan_profile("Electricity", 336, 720, 128, 512, 288, 48),
        
        pan_profile("ETTh1", 336, 720, 512, 512, 192, 48),
        pan_profile("ETTh2", 336, 720, 512, 512, 96, 48),
        pan_profile("ETTm1", 336, 720, 512, 512, 96, 48),
        pan_profile("ETTm2", 336, 720, 512, 512, 96, 48),
        pan_profile("Traffic", 336, 720, 512, 512, 192, 48),
        pan_profile("ExchangeRate", 336, 720, 512, 512, 96, 48),
        pan_profile("Weather", 336, 720, 512, 512, 96, 48),
        pan_profile("Electricity", 336, 720, 512, 512, 288, 48),
        
        # pan_profile("PEMS04", 96, 12, 128, 512, 96, 48),
        # pan_profile("PEMS07", 96, 12, 128, 512, 96, 48),
        # pan_profile("PEMS08", 96, 12, 128, 512, 96, 48),
        # pan_profile("PEMS_BAY", 96, 12, 128, 512, 96, 48),   
    ]
    print(all_data)
    df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "lowmem", "traintime", "parameters"])
    df.to_csv('traffic_pan_profile.csv')
    
# import fire
# fire.Fire(baseline_profile)


profile_pan_all()
# from torch_timeseries.experiments import *
# profile_pan_windows()


# profile_al_lpo()