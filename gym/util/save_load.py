import torch 
import torch.nn as nn
from pathlib import Path
from typing import Union

def load_ckpt(model:nn.Module, ckpt_path:str, optimizer:Union[torch.optim.Optimizer, None]):
    assert Path(ckpt_path).exists(), f"ckpt not found: {ckpt_path}"
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("agent model loaded!")
    return model, optimizer, start_epoch

def save_ckpt(model:nn.Module, folder_path:str, optimizer:torch.optim.Optimizer, epoch:int=0):
    assert Path(folder_path).exists(), f"folder not found: {folder_path}"
    ckpt_path = Path(folder_path, "agent_model.pth")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_reward': 0.0,
        'epoch': epoch  # 当前训练的轮数
    }
    torch.save(checkpoint, ckpt_path)
    print("agent model saved!")