import json
#from train_model import create_model
import configparser
import traceback

from augment import call_augmentation_functions

def main():

    config_path = "../config/config.json"
    cfg = json.load(open(config_path, 'r'))

    config = configparser.ConfigParser()
    config.read('../config/config.ini')
    aug_types = {'blur': config['AUGMENTATION']['Blur'], \
                 'bright': config['AUGMENTATION']['Bright'], \
                 'warp': config['AUGMENTATION']['Warp'], \
                 'rotate': config['AUGMENTATION']['rotate'], \
                 'flip': config['AUGMENTATION']['Flip']
                 }
    data_path = config['DATASET_PATH']['images_path']
    call_augmentation_functions(aug_types, data_path, config)

    '''data_dir = cfg['data_dir']
    model_obj = create_model([cfg['experiment_name'], cfg['model_arch'], cfg['num_classes'],data_dir , cfg['train_params']])
    model_obj.make_model()'''

if __name__ == "__main__":
    #print("*"*70)
    main()
