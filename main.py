import argparse
import yaml
import numpy as np

from model import InceptionTime, Transformer
from data_util import create_dataloader
from train_util import Trainer



def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def load_data(path,config):
    data = np.load(path,allow_pickle=True).item()
    
    sequence,label = data['data'], data['label']
    B,C,T = sequence.shape

    assert config['dataset']['seq_len'] == T and config['dataset']['feature_size'] == C
    
    train_loader , test_loader = create_dataloader(sequence,label,
                                                   config['dataset']['batch_size'],config['dataset']['split_size'],config['dataset']['label_type'])
    

    return  train_loader, test_loader


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default=None)
    parser.add_argument('--ckpt',type=str,default=None)
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--config',type=str,default=None)
    parser.add_argument('--model',type=str)

    

    args = parser.parse_args()

    return args


def main():
    args = parse_argument()
    config = load_yaml_config(args.config)

    if args.model == 'Inception':
        model = InceptionTime(config['dataset']['seq_len'],config['dataset']['feature_size'],config['dataset']['label_dim'],
                            filter_size=config['model']['filter_size'],dropout=config['model']['dropout'],depth=config['model']['depth'],kernels=config['model']['kernels'],
                            use_residual=config['model']['use_residual'],use_bottleneck=config['model']['use_bottleneck'],use_attn=config['model']['use_attn'],use_embedding=config['model']['use_embedding'])
    
    elif args.model == 'Transformer':
        model = Transformer(config['dataset']['seq_len'],config['dataset']['feature_size'],config['dataset']['label_dim'],
                            d_model = config['model']['d_model'],n_head = config['model']['n_head'],fn_hidden=config['model']['fn_hidden'],
                            n_layers = config['model']['n_layers'],dropout = config['model']['dropout'])

    train_dataloader , test_dataloader = load_data(args.data,config)
    trainer = Trainer(model,config['dataset']['max_iteration'],config['dataset']['lr'],config['dataset']['save_iteration'],args.ckpt)

    trainer.fit(train_dataloader,test_dataloader)


if __name__ == '__main__':
    main()


    

