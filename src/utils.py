import yaml
import torch
import torch.cuda
import torch.backends.cudnn
import random
import numpy as np


def read_options():
    with open('config.yml', 'r') as f:
        params = yaml.load(f.read(), Loader=yaml.Loader)
    if 'Round' not in params['Trainer']:
        params['Trainer']['Round'] = params['Trainer']['total_epoch'] // params['Trainer']['E']
    return params
