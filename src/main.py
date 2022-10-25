import json
import os
import numpy as np
import sys
from train_model import create_model

def main():

    config_path = "../config/config.json"
    cfg = json.load(open(config_path,'r'))

    data_dir = cfg['data_dir']

    model_obj = create_model([cfg['experiment_name'], cfg['model_arch'], cfg['num_classes'],data_dir , cfg['train_params']])
    model_obj.make_model()

if __name__ == "__main__":
    print("*"*70)
    main()
