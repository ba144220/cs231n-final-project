from termcolor import cprint
import torch
import torch.nn as nn
import pathlib
import dill
def print_params(model):
    """
    Print the number of parameters in each part of the model.
    """    
    params_dict = {}

    all_num_param = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        part_name = name.split('.')[0]
        if part_name not in params_dict:
            params_dict[part_name] = 0
        params_dict[part_name] += param.numel()
        
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    cprint(f'----------------------------------', 'cyan')
    cprint(f'Class name: {model.__class__.__name__}', 'cyan')
    cprint(f'  Number of parameters: {all_num_param / 1e6:.4f}M', 'cyan')
    for part_name, num_params in params_dict.items():
        cprint(f'   {part_name}: {num_params / 1e6:.4f}M ({num_params / all_num_param:.2%})', 'cyan')
        
    cprint(f'  Number of frozen parameters: {frozen_params / 1e6:.4f}M', 'cyan')
    cprint(f'  Percentage of frozen parameters: {frozen_params / all_num_param:.2%}', 'cyan')
    cprint(f'----------------------------------', 'cyan')
    
    
def save_submodule(
    submodule: nn.Module,
    path: pathlib.Path,
):
    """
    Save the submodule of the model.
    """
    torch.save(submodule.state_dict(), path.open('wb'), pickle_module=dill)
    cprint(f'Submodule saved to {path}', 'green')
    
def load_submodule(
    submodule: nn.Module,
    path: pathlib.Path,
):
    """
    Load the submodule of the model.
    """
    submodule.load_state_dict(torch.load(path.open('rb'), pickle_module=dill))
    cprint(f'Submodule loaded from {path}', 'green')

def freeze_submodule(submodule: nn.Module):
    """
    Freeze the submodule of the model.
    """
    for param in submodule.parameters():
        param.requires_grad = False
        
    cprint(f'Submodule {submodule.__class__.__name__} frozen', 'red')
