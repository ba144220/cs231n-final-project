"""
Extract the obs_encoder from the model checkpoint
Example:

python extract_obs_encoder.py --ckpt_path /path/to/model.ckpt

This will save the obs_encoder to a new file called model_obs_encoder.ckpt
"""

import torch
import dill
import argparse
from termcolor import cprint
import pathlib
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()

    output_path = args.ckpt_path.replace('.ckpt', '_obs_encoder.ckpt')
    output_path = pathlib.Path(output_path)
    
    ckpt = torch.load(args.ckpt_path, pickle_module=dill)

    model_state_dict = ckpt['state_dicts']['model']
    
    obs_encoder_state_dict = {}
            
    # Extract obs_encoder parameters
    for param_name, param_value in model_state_dict.items():
        if param_name.startswith('obs_encoder.'):
            obs_encoder_state_dict[param_name] = param_value
            
    if obs_encoder_state_dict:
        torch.save(obs_encoder_state_dict, output_path.open('wb'), pickle_module=dill)
        cprint(f'Saved obs_encoder to {output_path}', 'green')
    else:
        cprint('No obs_encoder found in the model', 'red')

if __name__ == '__main__':
    main()



